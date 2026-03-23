#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::fs;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full, StreamBody};
    use hyper::body::{Frame, Incoming};
    use hyper::service::service_fn;
    use hyper::{Method, Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use semantic_safety_protocol::{SemanticEntity, SemanticSafetyServiceServer, SemanticTopic};
    use semantic_safety_service::backend::TensorRtBackend;
    use semantic_safety_service::persistence::FileProjectIndexStore;
    use semantic_safety_service::service::{
        SemanticSafetyConfig as ServiceConfig, SemanticSafetyGrpcService,
    };
    use tempfile::NamedTempFile;
    use tempfile::TempDir;
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;
    use tokio::time::sleep;
    use tokio_stream::wrappers::UnboundedReceiverStream;
    use tokio_stream::StreamExt;
    use tonic::transport::Server;

    use plugin_llm_gateway::store::{
        GatewayStore, ProjectSemanticPolicyRecord, ProjectToolRecord, RequestLogEntry,
        SafetyPolicyRecord,
    };
    use plugin_llm_gateway::CreatePluginsOptions;
    use proxy_core::config::{
        PluginConfig, ProviderFamilyConfig, ProviderKeyConfig, ProviderSurfaceCatalog,
        ProviderToolProtocol, ResponsesSurface, ToolSurface,
    };
    use proxy_core::plugin::PluginChain;

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream_async,
        TestProxyConfig,
    };

    type McpServerConfig<'a> = (&'a str, &'a str, Option<u64>, Option<u32>, Option<u32>);
    type McpServerTimeBudgetConfig<'a> = (
        &'a str,
        &'a str,
        Option<u64>,
        Option<u32>,
        Option<u32>,
        Option<u64>,
    );
    type McpServerOutputBudgetConfig<'a> =
        (&'a str, &'a str, Option<u64>, Option<u32>, Option<u64>);

    fn tool_runtime_config() -> Vec<PluginConfig> {
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(toml::value::Map::new()),
        }]
    }

    fn tool_runtime_config_with_responses_stream_mode(mode: &str) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        config.insert(
            "responses_stream_mode".into(),
            toml::Value::String(mode.to_string()),
        );
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_backends(
        web_search_backends: &[(&str, &str)],
        arxiv_base_url: Option<&str>,
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        if !web_search_backends.is_empty() {
            let mut backends = toml::value::Map::new();
            for (name, url) in web_search_backends {
                let mut backend = toml::value::Map::new();
                backend.insert("url".into(), toml::Value::String((*url).to_string()));
                backend.insert("method".into(), toml::Value::String("POST".to_string()));
                backends.insert((*name).to_string(), toml::Value::Table(backend));
            }
            config.insert("web_search_backends".into(), toml::Value::Table(backends));
        }
        if let Some(base_url) = arxiv_base_url {
            config.insert(
                "arxiv_base_url".into(),
                toml::Value::String(base_url.to_string()),
            );
        }
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_mcp_server_options(
        mcp_servers: &[(&str, &str, Option<u64>, Option<u32>)],
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        if !mcp_servers.is_empty() {
            let mut servers = toml::value::Map::new();
            for (name, url, timeout_ms, max_retries) in mcp_servers {
                let mut server = toml::value::Map::new();
                server.insert("url".into(), toml::Value::String((*url).to_string()));
                server.insert("method".into(), toml::Value::String("POST".to_string()));
                if let Some(timeout_ms) = timeout_ms {
                    server.insert(
                        "timeout_ms".into(),
                        toml::Value::Integer(*timeout_ms as i64),
                    );
                }
                if let Some(max_retries) = max_retries {
                    server.insert(
                        "max_retries".into(),
                        toml::Value::Integer(*max_retries as i64),
                    );
                }
                servers.insert((*name).to_string(), toml::Value::Table(server));
            }
            config.insert("mcp_servers".into(), toml::Value::Table(servers));
        }
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_mcp_server_budget_options(
        mcp_servers: &[McpServerConfig<'_>],
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        if !mcp_servers.is_empty() {
            let mut servers = toml::value::Map::new();
            for (name, url, timeout_ms, max_retries, max_calls_per_request) in mcp_servers {
                let mut server = toml::value::Map::new();
                server.insert("url".into(), toml::Value::String((*url).to_string()));
                server.insert("method".into(), toml::Value::String("POST".to_string()));
                if let Some(timeout_ms) = timeout_ms {
                    server.insert(
                        "timeout_ms".into(),
                        toml::Value::Integer(*timeout_ms as i64),
                    );
                }
                if let Some(max_retries) = max_retries {
                    server.insert(
                        "max_retries".into(),
                        toml::Value::Integer(*max_retries as i64),
                    );
                }
                if let Some(max_calls_per_request) = max_calls_per_request {
                    server.insert(
                        "max_calls_per_request".into(),
                        toml::Value::Integer(*max_calls_per_request as i64),
                    );
                }
                servers.insert((*name).to_string(), toml::Value::Table(server));
            }
            config.insert("mcp_servers".into(), toml::Value::Table(servers));
        }
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_mcp_server_time_budget_options(
        mcp_servers: &[McpServerTimeBudgetConfig<'_>],
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        if !mcp_servers.is_empty() {
            let mut servers = toml::value::Map::new();
            for (name, url, timeout_ms, max_retries, max_calls_per_request, max_total_time_ms) in
                mcp_servers
            {
                let mut server = toml::value::Map::new();
                server.insert("url".into(), toml::Value::String((*url).to_string()));
                server.insert("method".into(), toml::Value::String("POST".to_string()));
                if let Some(timeout_ms) = timeout_ms {
                    server.insert(
                        "timeout_ms".into(),
                        toml::Value::Integer(*timeout_ms as i64),
                    );
                }
                if let Some(max_retries) = max_retries {
                    server.insert(
                        "max_retries".into(),
                        toml::Value::Integer(*max_retries as i64),
                    );
                }
                if let Some(max_calls_per_request) = max_calls_per_request {
                    server.insert(
                        "max_calls_per_request".into(),
                        toml::Value::Integer(*max_calls_per_request as i64),
                    );
                }
                if let Some(max_total_time_ms) = max_total_time_ms {
                    server.insert(
                        "max_total_time_ms".into(),
                        toml::Value::Integer(*max_total_time_ms as i64),
                    );
                }
                servers.insert((*name).to_string(), toml::Value::Table(server));
            }
            config.insert("mcp_servers".into(), toml::Value::Table(servers));
        }
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_mcp_server_output_budget_options(
        mcp_servers: &[McpServerOutputBudgetConfig<'_>],
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        if !mcp_servers.is_empty() {
            let mut servers = toml::value::Map::new();
            for (name, url, timeout_ms, max_retries, max_output_tokens) in mcp_servers {
                let mut server = toml::value::Map::new();
                server.insert("url".into(), toml::Value::String((*url).to_string()));
                server.insert("method".into(), toml::Value::String("POST".to_string()));
                if let Some(timeout_ms) = timeout_ms {
                    server.insert(
                        "timeout_ms".into(),
                        toml::Value::Integer(*timeout_ms as i64),
                    );
                }
                if let Some(max_retries) = max_retries {
                    server.insert(
                        "max_retries".into(),
                        toml::Value::Integer(*max_retries as i64),
                    );
                }
                if let Some(max_output_tokens) = max_output_tokens {
                    server.insert(
                        "max_output_tokens".into(),
                        toml::Value::Integer(*max_output_tokens as i64),
                    );
                }
                servers.insert((*name).to_string(), toml::Value::Table(server));
            }
            config.insert("mcp_servers".into(), toml::Value::Table(servers));
        }
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_mcp_servers(mcp_servers: &[(&str, &str)]) -> Vec<PluginConfig> {
        let servers = mcp_servers
            .iter()
            .map(|(name, url)| (*name, *url, None, None))
            .collect::<Vec<_>>();
        tool_runtime_config_with_mcp_server_options(&servers)
    }

    fn tool_runtime_config_with_oauth_mcp_server(
        name: &str,
        url: &str,
        token_url: &str,
        client_id: &str,
        client_secret: &str,
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        let mut servers = toml::value::Map::new();
        let mut server = toml::value::Map::new();
        server.insert("url".into(), toml::Value::String(url.to_string()));
        server.insert("method".into(), toml::Value::String("POST".to_string()));
        server.insert(
            "auth".into(),
            toml::Value::Table({
                let mut auth = toml::value::Map::new();
                auth.insert(
                    "type".into(),
                    toml::Value::String("oauth_client_credentials".to_string()),
                );
                auth.insert(
                    "token_url".into(),
                    toml::Value::String(token_url.to_string()),
                );
                auth.insert(
                    "client_id".into(),
                    toml::Value::String(client_id.to_string()),
                );
                auth.insert(
                    "client_secret".into(),
                    toml::Value::String(client_secret.to_string()),
                );
                auth
            }),
        );
        servers.insert(name.to_string(), toml::Value::Table(server));
        config.insert("mcp_servers".into(), toml::Value::Table(servers));
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_discovered_oauth_mcp_server(
        name: &str,
        url: &str,
        client_id: &str,
        client_secret: &str,
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        let mut servers = toml::value::Map::new();
        let mut server = toml::value::Map::new();
        server.insert("url".into(), toml::Value::String(url.to_string()));
        server.insert("method".into(), toml::Value::String("POST".to_string()));
        server.insert(
            "auth".into(),
            toml::Value::Table({
                let mut auth = toml::value::Map::new();
                auth.insert(
                    "type".into(),
                    toml::Value::String("oauth_client_credentials".to_string()),
                );
                auth.insert(
                    "client_id".into(),
                    toml::Value::String(client_id.to_string()),
                );
                auth.insert(
                    "client_secret".into(),
                    toml::Value::String(client_secret.to_string()),
                );
                auth
            }),
        );
        servers.insert(name.to_string(), toml::Value::Table(server));
        config.insert("mcp_servers".into(), toml::Value::Table(servers));
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_oauth_authorization_code_mcp_server(
        name: &str,
        url: &str,
        client_id: &str,
        redirect_uri: &str,
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        let mut servers = toml::value::Map::new();
        let mut server = toml::value::Map::new();
        server.insert("url".into(), toml::Value::String(url.to_string()));
        server.insert("method".into(), toml::Value::String("POST".to_string()));
        server.insert(
            "auth".into(),
            toml::Value::Table({
                let mut auth = toml::value::Map::new();
                auth.insert(
                    "type".into(),
                    toml::Value::String("oauth_authorization_code".to_string()),
                );
                auth.insert(
                    "client_id".into(),
                    toml::Value::String(client_id.to_string()),
                );
                auth.insert(
                    "redirect_uri".into(),
                    toml::Value::String(redirect_uri.to_string()),
                );
                auth
            }),
        );
        servers.insert(name.to_string(), toml::Value::Table(server));
        config.insert("mcp_servers".into(), toml::Value::Table(servers));
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_stdio_mcp_server(
        name: &str,
        command: &str,
        args: &[String],
        env: &[(String, String)],
    ) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        let mut servers = toml::value::Map::new();
        let mut server = toml::value::Map::new();
        server.insert("transport".into(), toml::Value::String("stdio".to_string()));
        server.insert("command".into(), toml::Value::String(command.to_string()));
        server.insert(
            "args".into(),
            toml::Value::Array(
                args.iter()
                    .cloned()
                    .map(toml::Value::String)
                    .collect::<Vec<_>>(),
            ),
        );
        if !env.is_empty() {
            let mut env_table = toml::value::Map::new();
            for (key, value) in env {
                env_table.insert(key.clone(), toml::Value::String(value.clone()));
            }
            server.insert("env".into(), toml::Value::Table(env_table));
        }
        servers.insert(name.to_string(), toml::Value::Table(server));
        config.insert("mcp_servers".into(), toml::Value::Table(servers));
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
    }

    fn tool_runtime_config_with_sse_mcp_server(name: &str, url: &str) -> Vec<PluginConfig> {
        let mut config = toml::value::Map::new();
        let mut servers = toml::value::Map::new();
        let mut server = toml::value::Map::new();
        server.insert("transport".into(), toml::Value::String("sse".to_string()));
        server.insert("url".into(), toml::Value::String(url.to_string()));
        servers.insert(name.to_string(), toml::Value::Table(server));
        config.insert("mcp_servers".into(), toml::Value::Table(servers));
        vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table(config),
        }]
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
                                (method, "/sse") if method == hyper::Method::GET => {
                                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                                    tx.send("event: endpoint\ndata: /messages\n\n".to_string())
                                        .unwrap();
                                    *active_sender.lock().await = Some(tx);
                                    let stream = UnboundedReceiverStream::new(rx).map(|chunk| {
                                        Ok::<_, Infallible>(Frame::data(Bytes::from(chunk)))
                                    });
                                    let body = StreamBody::new(stream);
                                    let response = Response::builder()
                                        .status(StatusCode::OK)
                                        .header("content-type", "text/event-stream")
                                        .body(body.boxed())
                                        .unwrap();
                                    Ok::<_, hyper::Error>(response)
                                }
                                (method, "/messages") if method == hyper::Method::POST => {
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
                                            "tools/call" => serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": body_json["id"].clone(),
                                                "result": {
                                                    "content": [{
                                                        "type": "text",
                                                        "text": format!(
                                                            "Remote MCP result: {}",
                                                            body_json["params"]["arguments"]["query"]
                                                                .as_str()
                                                                .unwrap_or_default()
                                                        )
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
                                    let response = Response::builder()
                                        .status(StatusCode::ACCEPTED)
                                        .body(Full::new(Bytes::new()).boxed())
                                        .unwrap();
                                    Ok::<_, hyper::Error>(response)
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

    fn plugin_configs(extra: Vec<PluginConfig>) -> Vec<PluginConfig> {
        let mut configs = tool_runtime_config();
        configs.extend(extra);
        configs
    }

    fn content_filter_config(action: &str) -> PluginConfig {
        PluginConfig {
            name: "content_filter".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("action".into(), toml::Value::String(action.to_string()));
                t
            }),
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

    fn semantic_safety_config(endpoint: &str) -> PluginConfig {
        PluginConfig {
            name: "semantic_safety".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("endpoint".into(), toml::Value::String(endpoint.to_string()));
                t.insert("timeout_ms".into(), toml::Value::Integer(200));
                t.insert("reconcile_interval_secs".into(), toml::Value::Integer(3600));
                t
            }),
        }
    }

    fn webhook_tool_record(project_id: &str, tool_name: &str, url: &str) -> ProjectToolRecord {
        ProjectToolRecord {
            project_id: project_id.to_string(),
            tool_name: tool_name.to_string(),
            description: Some("Test tool".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "webhook".to_string(),
            executor_config_json: Some(
                serde_json::json!({
                    "url": url,
                    "method": "POST"
                })
                .to_string(),
            ),
            enabled: true,
            timeout_ms: Some(2_000),
            updated_at: "0".to_string(),
        }
    }

    fn web_search_tool_record(
        project_id: &str,
        tool_name: &str,
        backend: &str,
    ) -> ProjectToolRecord {
        ProjectToolRecord {
            project_id: project_id.to_string(),
            tool_name: tool_name.to_string(),
            description: Some("Search the web".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "web_search".to_string(),
            executor_config_json: Some(
                serde_json::json!({
                    "backend": backend
                })
                .to_string(),
            ),
            enabled: true,
            timeout_ms: Some(2_000),
            updated_at: "0".to_string(),
        }
    }

    fn arxiv_tool_record(project_id: &str, tool_name: &str, max_results: u64) -> ProjectToolRecord {
        ProjectToolRecord {
            project_id: project_id.to_string(),
            tool_name: tool_name.to_string(),
            description: Some("Search arXiv".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "arxiv_search".to_string(),
            executor_config_json: Some(
                serde_json::json!({
                    "max_results": max_results
                })
                .to_string(),
            ),
            enabled: true,
            timeout_ms: Some(2_000),
            updated_at: "0".to_string(),
        }
    }

    fn mcp_tool_record(
        project_id: &str,
        tool_name: &str,
        server: &str,
        remote_tool: &str,
    ) -> ProjectToolRecord {
        ProjectToolRecord {
            project_id: project_id.to_string(),
            tool_name: tool_name.to_string(),
            description: Some("Call a remote MCP tool".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "mcp".to_string(),
            executor_config_json: Some(
                serde_json::json!({
                    "server": server,
                    "remote_tool": remote_tool
                })
                .to_string(),
            ),
            enabled: true,
            timeout_ms: Some(2_000),
            updated_at: "0".to_string(),
        }
    }

    fn mcp_tool_record_with_budget(
        project_id: &str,
        tool_name: &str,
        server: &str,
        remote_tool: &str,
        max_calls_per_request: u32,
    ) -> ProjectToolRecord {
        ProjectToolRecord {
            project_id: project_id.to_string(),
            tool_name: tool_name.to_string(),
            description: Some("Call a remote MCP tool".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "mcp".to_string(),
            executor_config_json: Some(
                serde_json::json!({
                    "server": server,
                    "remote_tool": remote_tool,
                    "max_calls_per_request": max_calls_per_request
                })
                .to_string(),
            ),
            enabled: true,
            timeout_ms: Some(2_000),
            updated_at: "0".to_string(),
        }
    }

    fn mcp_tool_record_with_time_budget(
        project_id: &str,
        tool_name: &str,
        server: &str,
        remote_tool: &str,
        max_total_time_ms: u64,
    ) -> ProjectToolRecord {
        ProjectToolRecord {
            project_id: project_id.to_string(),
            tool_name: tool_name.to_string(),
            description: Some("Call a remote MCP tool".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "mcp".to_string(),
            executor_config_json: Some(
                serde_json::json!({
                    "server": server,
                    "remote_tool": remote_tool,
                    "max_total_time_ms": max_total_time_ms
                })
                .to_string(),
            ),
            enabled: true,
            timeout_ms: Some(2_000),
            updated_at: "0".to_string(),
        }
    }

    fn mcp_tool_record_with_output_budget(
        project_id: &str,
        tool_name: &str,
        server: &str,
        remote_tool: &str,
        max_output_tokens: u64,
    ) -> ProjectToolRecord {
        ProjectToolRecord {
            project_id: project_id.to_string(),
            tool_name: tool_name.to_string(),
            description: Some("Call a remote MCP tool".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "mcp".to_string(),
            executor_config_json: Some(
                serde_json::json!({
                    "server": server,
                    "remote_tool": remote_tool,
                    "max_output_tokens": max_output_tokens
                })
                .to_string(),
            ),
            enabled: true,
            timeout_ms: Some(2_000),
            updated_at: "0".to_string(),
        }
    }

    fn semantic_policy_record(project_id: &str) -> ProjectSemanticPolicyRecord {
        ProjectSemanticPolicyRecord {
            project_id: project_id.to_string(),
            version: "v1".to_string(),
            enabled: true,
            entities_json: Some(
                serde_json::to_string(&vec![SemanticEntity {
                    entity_id: "company-x".to_string(),
                    name: "Company X".to_string(),
                    aliases: vec!["companyx".to_string()],
                }])
                .unwrap(),
            ),
            topics_json: Some(
                serde_json::to_string(&vec![SemanticTopic {
                    topic_id: "layoffs".to_string(),
                    name: "Layoffs".to_string(),
                    exemplars: vec!["company x layoffs next week".to_string()],
                    rerank_threshold: 0.1,
                    require_entity_match: true,
                }])
                .unwrap(),
            ),
            updated_at: "0".to_string(),
        }
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

    fn tool_surface(protocol: ProviderToolProtocol) -> Option<ToolSurface> {
        match protocol {
            ProviderToolProtocol::None => None,
            ProviderToolProtocol::OpenAi => Some(ToolSurface::OpenAi),
            ProviderToolProtocol::Anthropic => Some(ToolSurface::Anthropic),
        }
    }

    fn openai_provider(base_url: String, protocol: ProviderToolProtocol) -> ProviderKeyConfig {
        provider(
            "openai",
            "sk-openai-real",
            base_url,
            vec!["gpt-4o".to_string()],
            "authorization",
            ProviderFamilyConfig::OpenAi {
                surfaces: ProviderSurfaceCatalog {
                    tools: tool_surface(protocol),
                    ..ProviderSurfaceCatalog::default()
                },
            },
        )
    }

    fn openai_responses_provider(
        base_url: String,
        protocol: ProviderToolProtocol,
    ) -> ProviderKeyConfig {
        provider(
            "openai",
            "sk-openai-real",
            base_url,
            vec!["gpt-4o".to_string()],
            "authorization",
            ProviderFamilyConfig::OpenAi {
                surfaces: ProviderSurfaceCatalog {
                    tools: tool_surface(protocol),
                    responses: Some(ResponsesSurface::OpenAiCompatible),
                    ..ProviderSurfaceCatalog::default()
                },
            },
        )
    }

    fn anthropic_provider(base_url: String, protocol: ProviderToolProtocol) -> ProviderKeyConfig {
        provider(
            "anthropic",
            "sk-anthropic-real",
            base_url,
            vec!["claude-sonnet-4-20250514".to_string()],
            "x-api-key",
            ProviderFamilyConfig::Anthropic {
                surfaces: ProviderSurfaceCatalog {
                    tools: tool_surface(protocol),
                    ..ProviderSurfaceCatalog::default()
                },
            },
        )
    }

    async fn setup_gateway(
        providers: &[ProviderKeyConfig],
    ) -> (Arc<PluginChain>, plugin_llm_gateway::api::LlmGatewayApi) {
        setup_gateway_with_configs(tool_runtime_config(), providers).await
    }

    async fn setup_gateway_with_configs(
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

    async fn start_semantic_service() -> (String, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap());
        let backend = Arc::new(TensorRtBackend::new_dev_stub());
        let service =
            SemanticSafetyGrpcService::new(ServiceConfig { auth_token: None }, store, backend);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
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

    async fn wait_for_request_logs(
        api: &plugin_llm_gateway::api::LlmGatewayApi,
        project_id: &str,
    ) -> Vec<RequestLogEntry> {
        for _ in 0..20 {
            if let Some(Ok(logs)) = api.get_request_logs(None, None, Some(project_id), 10).await {
                if !logs.is_empty() {
                    return logs;
                }
            }
            sleep(Duration::from_millis(25)).await;
        }
        Vec::new()
    }

    #[tokio::test]
    async fn openai_request_uses_registered_webhook_tool() {
        let project_id = "project-openai";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_requests = Arc::new(Mutex::new(Vec::new()));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            let tool_requests = Arc::clone(&tool_requests);
            move |req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                let tool_requests = Arc::clone(&tool_requests);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect tool body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("tool request json");
                    tool_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"Rust RFC paper"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_requests = Arc::new(Mutex::new(Vec::new()));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            let provider_requests = Arc::clone(&provider_requests);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                let provider_requests = Arc::clone(&provider_requests);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });
                    provider_requests.lock().await.push(body_json.clone());

                    if call_index == 0 {
                        assert!(body_json.get("trp_tools").is_none());
                        let tool_names = body_json["tools"]
                            .as_array()
                            .expect("tools array")
                            .iter()
                            .filter_map(|tool| {
                                tool.get("function")
                                    .and_then(|value| value.get("name"))
                                    .and_then(|value| value.as_str())
                            })
                            .collect::<Vec<_>>();
                        assert!(tool_names.contains(&"web_search"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-tool-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "web_search",
                                                    "arguments": "{\"query\":\"rust rfc\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 12,
                                        "completion_tokens": 4,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Rust RFC paper"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Found the Rust RFC paper via the tool."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 18,
                                        "completion_tokens": 7,
                                        "total_tokens": 25
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Found the Rust RFC paper via the tool.")
        );
        assert_eq!(tool_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let tool_request = tool_requests.lock().await;
        assert_eq!(tool_request.len(), 1);
        assert_eq!(
            tool_request[0]["arguments"]["query"].as_str(),
            Some("rust rfc")
        );
    }

    #[tokio::test]
    async fn openai_responses_request_uses_registered_webhook_tool() {
        let project_id = "project-openai-responses";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_requests = Arc::new(Mutex::new(Vec::new()));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            let tool_requests = Arc::clone(&tool_requests);
            move |req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                let tool_requests = Arc::clone(&tool_requests);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect tool body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("tool request json");
                    tool_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"Rust RFC paper"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_requests = Arc::new(Mutex::new(Vec::new()));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            let provider_requests = Arc::clone(&provider_requests);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                let provider_requests = Arc::clone(&provider_requests);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    assert_eq!(req.uri().path(), "/v1/responses");
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });
                    provider_requests.lock().await.push(body_json.clone());

                    if call_index == 0 {
                        assert!(body_json.get("trp_tools").is_none());
                        let tool_names = body_json["tools"]
                            .as_array()
                            .expect("tools array")
                            .iter()
                            .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
                            .collect::<Vec<_>>();
                        assert!(tool_names.contains(&"web_search"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "resp_tool_1",
                                    "object": "response",
                                    "model": "gpt-4o",
                                    "status": "completed",
                                    "output": [{
                                        "type": "function_call",
                                        "id": "fc_1",
                                        "call_id": "call_1",
                                        "name": "web_search",
                                        "arguments": "{\"query\":\"rust rfc\"}",
                                        "status": "completed"
                                    }],
                                    "usage": {
                                        "input_tokens": 12,
                                        "output_tokens": 4,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        assert_eq!(
                            body_json["previous_response_id"].as_str(),
                            Some("resp_tool_1")
                        );
                        let input = body_json["input"].as_array().expect("input array");
                        assert_eq!(input.len(), 1);
                        assert_eq!(input[0]["type"].as_str(), Some("function_call_output"));
                        assert_eq!(input[0]["call_id"].as_str(), Some("call_1"));
                        assert!(input[0]["output"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Rust RFC paper"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "resp_tool_2",
                                    "object": "response",
                                    "model": "gpt-4o",
                                    "status": "completed",
                                    "output": [{
                                        "type": "message",
                                        "id": "msg_1",
                                        "status": "completed",
                                        "role": "assistant",
                                        "content": [{
                                            "type": "output_text",
                                            "text": "Found the Rust RFC paper via the tool."
                                        }]
                                    }],
                                    "usage": {
                                        "input_tokens": 18,
                                        "output_tokens": 7,
                                        "total_tokens": 25
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_responses_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
            .uri("/v1/responses")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "input": [{
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": "find the paper"
                        }]
                    }],
                    "tool_choice": {
                        "type": "allowed_tools",
                        "tools": ["web_search"]
                    },
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["output"][0]["content"][0]["text"].as_str(),
            Some("Found the Rust RFC paper via the tool.")
        );
        assert_eq!(tool_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let tool_request = tool_requests.lock().await;
        assert_eq!(tool_request.len(), 1);
        assert_eq!(
            tool_request[0]["arguments"]["query"].as_str(),
            Some("rust rfc")
        );
    }

    #[tokio::test]
    async fn anthropic_request_uses_registered_webhook_tool() {
        let project_id = "project-anthropic";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            move |req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect tool body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("tool request json");
                    assert_eq!(
                        body_json["arguments"]["query"].as_str(),
                        Some("attention is all you need")
                    );
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"arXiv:1706.03762"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });

                    if call_index == 0 {
                        assert!(body_json.get("trp_tools").is_none());
                        let tools = body_json["tools"].as_array().expect("tools");
                        assert!(tools.iter().any(|tool| {
                            tool.get("name").and_then(|value| value.as_str())
                                == Some("arxiv_search")
                        }));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "msg_1",
                                    "type": "message",
                                    "role": "assistant",
                                    "model": "claude-sonnet-4-20250514",
                                    "content": [{
                                        "type": "tool_use",
                                        "id": "toolu_1",
                                        "name": "arxiv_search",
                                        "input": {
                                            "query": "attention is all you need"
                                        }
                                    }],
                                    "stop_reason": "tool_use",
                                    "usage": {
                                        "input_tokens": 15,
                                        "output_tokens": 5
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[2]["role"].as_str(), Some("user"));
                        assert!(messages[2]["content"][0]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("1706.03762"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "msg_2",
                                    "type": "message",
                                    "role": "assistant",
                                    "model": "claude-sonnet-4-20250514",
                                    "content": [{
                                        "type": "text",
                                        "text": "The paper is arXiv:1706.03762."
                                    }],
                                    "stop_reason": "end_turn",
                                    "usage": {
                                        "input_tokens": 19,
                                        "output_tokens": 8
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![anthropic_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::Anthropic,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "arxiv_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "max_tokens": 256,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": "find the paper"
                        }]
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["arxiv_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["content"][0]["text"].as_str(),
            Some("The paper is arXiv:1706.03762.")
        );
        assert_eq!(tool_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn openai_request_uses_remote_mcp_tool() {
        let project_id = "project-openai-mcp";
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
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
                            assert_eq!(body_json["jsonrpc"].as_str(), Some("2.0"));
                            assert_eq!(
                                body_json["params"]["protocolVersion"].as_str(),
                                Some("2025-11-25")
                            );
                            return Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .header("mcp-session-id", "session-test-1")
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
                                .unwrap();
                        }
                        Some("notifications/initialized") => {
                            assert_eq!(
                                headers
                                    .get("mcp-session-id")
                                    .and_then(|value| value.to_str().ok()),
                                Some("session-test-1")
                            );
                            assert_eq!(
                                headers
                                    .get("mcp-protocol-version")
                                    .and_then(|value| value.to_str().ok()),
                                Some("2025-11-25")
                            );
                            return Response::builder()
                                .status(StatusCode::ACCEPTED)
                                .body(Full::new(Bytes::new()))
                                .unwrap();
                        }
                        Some("tools/list") => {
                            assert_eq!(
                                headers
                                    .get("mcp-session-id")
                                    .and_then(|value| value.to_str().ok()),
                                Some("session-test-1")
                            );
                            assert_eq!(
                                headers
                                    .get("mcp-protocol-version")
                                    .and_then(|value| value.to_str().ok()),
                                Some("2025-11-25")
                            );
                            return Response::builder()
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
                                .unwrap();
                        }
                        Some("tools/call") => {
                            assert_eq!(
                                headers
                                    .get("mcp-session-id")
                                    .and_then(|value| value.to_str().ok()),
                                Some("session-test-1")
                            );
                            assert_eq!(
                                headers
                                    .get("mcp-protocol-version")
                                    .and_then(|value| value.to_str().ok()),
                                Some("2025-11-25")
                            );
                            assert_eq!(body_json["params"]["name"].as_str(), Some("search_docs"));
                            assert_eq!(
                                body_json["params"]["arguments"]["query"].as_str(),
                                Some("rust mcp")
                            );
                        }
                        other => panic!("unexpected MCP method: {:?}", other),
                    }
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "jsonrpc": "2.0",
                                "id": body_json["id"].clone(),
                                "result": {
                                    "content": [{
                                        "type": "text",
                                        "text": "Remote MCP result: Rust MCP paper"
                                    }]
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });

                    if call_index == 0 {
                        let tool_names = body_json["tools"]
                            .as_array()
                            .expect("tools array")
                            .iter()
                            .filter_map(|tool| {
                                tool.get("function")
                                    .and_then(|value| value.get("name"))
                                    .and_then(|value| value.as_str())
                            })
                            .collect::<Vec<_>>();
                        assert!(tool_names.contains(&"mcp_search"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 11,
                                        "completion_tokens": 5,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Remote MCP result: Rust MCP paper"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "MCP search found the Rust MCP paper."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 17,
                                        "completion_tokens": 6,
                                        "total_tokens": 23
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let mcp_url = format!("http://{}", mcp_addr);
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_mcp_servers(&[("docs", &mcp_url)]),
            &providers,
        )
        .await;
        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the MCP paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("MCP search found the Rust MCP paper.")
        );
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 4);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn openai_request_uses_oauth_authenticated_mcp_tool() {
        let project_id = "project-openai-mcp-oauth";
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
                    assert!(body_text.contains("client_id=tool-runtime"), "{body_text}");
                    assert!(body_text.contains("client_secret=topsecret"), "{body_text}");
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "access_token": "oauth-token-1",
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

        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
                    assert_eq!(
                        req.headers()
                            .get("authorization")
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer oauth-token-1")
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
                            .header("mcp-session-id", "session-oauth-1")
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
                                Some("session-oauth-1")
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
                        Some("tools/call") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "content": [{
                                            "type": "text",
                                            "text": "Remote MCP result: Rust MCP OAuth paper"
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

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider json");
                    if call_index == 0 {
                        let tool_names = body_json["tools"]
                            .as_array()
                            .expect("tools array")
                            .iter()
                            .filter_map(|tool| {
                                tool.get("function")
                                    .and_then(|value| value.get("name"))
                                    .and_then(|value| value.as_str())
                            })
                            .collect::<Vec<_>>();
                        assert!(tool_names.contains(&"mcp_search"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-oauth-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_oauth_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp oauth\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Remote MCP result: Rust MCP OAuth paper"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-oauth-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "MCP OAuth search found the Rust MCP paper."
                                        },
                                        "finish_reason": "stop"
                                    }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_oauth_mcp_server(
                "docs",
                &format!("http://{}", mcp_addr),
                &format!("http://{}", token_addr),
                "tool-runtime",
                "topsecret",
            ),
            &providers,
        )
        .await;
        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the MCP paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("MCP OAuth search found the Rust MCP paper.")
        );
        assert_eq!(token_hits.load(Ordering::Relaxed), 1);
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 4);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn openai_request_uses_stdio_mcp_tool() {
        let project_id = "project-openai-mcp-stdio";
        let log_file = NamedTempFile::new().expect("stdio log file");
        let log_path = log_file.path().to_string_lossy().to_string();
        let script = fake_stdio_mcp_script();
        let config = tool_runtime_config_with_stdio_mcp_server(
            "docs",
            &uv_bin(),
            &[
                "run".to_string(),
                "python".to_string(),
                "-u".to_string(),
                script,
            ],
            &[
                ("TRP_STDIO_MCP_LOG".to_string(), log_path.clone()),
                (
                    "TRP_STDIO_MCP_REMOTE_TOOL".to_string(),
                    "search_docs".to_string(),
                ),
            ],
        );

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });

                    if call_index == 0 {
                        let tool_names = body_json["tools"]
                            .as_array()
                            .expect("tools array")
                            .iter()
                            .filter_map(|tool| {
                                tool.get("function")
                                    .and_then(|value| value.get("name"))
                                    .and_then(|value| value.as_str())
                            })
                            .collect::<Vec<_>>();
                        assert!(tool_names.contains(&"mcp_search"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-stdio-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_stdio_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp stdio\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 11,
                                        "completion_tokens": 5,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Remote MCP result: rust mcp stdio"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-stdio-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "MCP stdio search found the Rust MCP paper."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 17,
                                        "completion_tokens": 6,
                                        "total_tokens": 23
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(config, &providers).await;
        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the MCP paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("MCP stdio search found the Rust MCP paper.")
        );
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);

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
                "tools/call"
            ]
        );
    }

    #[tokio::test]
    async fn openai_request_discovers_oauth_authenticated_mcp_tool() {
        let project_id = "project-openai-mcp-oauth-discovery";
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
                    assert!(body_text.contains("client_id=tool-runtime"), "{body_text}");
                    assert!(body_text.contains("client_secret=topsecret"), "{body_text}");
                    assert!(body_text.contains("resource=http%3A%2F%2F"), "{body_text}");
                    assert!(body_text.contains("%2Fmcp"), "{body_text}");
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "access_token": "oauth-discovered-token",
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
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let protected_metadata_hits = Arc::clone(&protected_metadata_hits);
            let mcp_hits = Arc::clone(&mcp_hits);
            let auth_addr = auth_addr.clone();
            move |req: Request<Incoming>| {
                let protected_metadata_hits = Arc::clone(&protected_metadata_hits);
                let mcp_hits = Arc::clone(&mcp_hits);
                let auth_addr = auth_addr.clone();
                async move {
                    let path = req.uri().path().to_string();
                    if req.method() == Method::GET
                        && path == "/.well-known/oauth-protected-resource/mcp"
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

                    assert_eq!(path, "/mcp");
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
                    assert_eq!(
                        req.headers()
                            .get("authorization")
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer oauth-discovered-token")
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
                            .header("mcp-session-id", "session-oauth-discovered")
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
                                Some("session-oauth-discovered")
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
                        Some("tools/call") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "content": [{
                                            "type": "text",
                                            "text": "Remote MCP result: OAuth discovery worked"
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

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider json");
                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-oauth-discovery-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_oauth_discovery_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp oauth discovery\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("OAuth discovery worked"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-oauth-discovery-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "MCP OAuth discovery search worked."
                                        },
                                        "finish_reason": "stop"
                                    }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_discovered_oauth_mcp_server(
                "docs",
                &format!("http://{}/mcp", mcp_addr),
                "tool-runtime",
                "topsecret",
            ),
            &providers,
        )
        .await;
        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the MCP paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("MCP OAuth discovery search worked.")
        );
        assert_eq!(protected_metadata_hits.load(Ordering::Relaxed), 1);
        assert_eq!(auth_metadata_hits.load(Ordering::Relaxed), 1);
        assert_eq!(token_hits.load(Ordering::Relaxed), 1);
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 4);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);

        let runtime_status = api.tool_runtime_status().expect("tool runtime status");
        let docs_server = runtime_status
            .mcp_servers
            .iter()
            .find(|server| server.name == "docs")
            .expect("docs mcp server");
        let expected_auth_url = format!("http://{auth_addr}");
        let expected_token_url = format!("http://{token_addr}/token");
        let expected_resource = format!("http://{mcp_addr}/mcp");
        assert_eq!(
            docs_server.auth_authorization_server_url.as_deref(),
            Some(expected_auth_url.as_str())
        );
        assert_eq!(
            docs_server.auth_token_url.as_deref(),
            Some(expected_token_url.as_str())
        );
        assert_eq!(
            docs_server.auth_resource.as_deref(),
            Some(expected_resource.as_str())
        );
        assert!(docs_server.auth_last_discovery_error.is_none());
    }

    #[tokio::test]
    async fn openai_request_uses_authorization_code_authenticated_mcp_tool() {
        let project_id = "project-openai-mcp-oauth-authcode";
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
                    assert!(body_text.contains("code=auth-code-1"), "{body_text}");
                    assert!(body_text.contains("code_verifier="), "{body_text}");
                    assert!(
                        body_text.contains("redirect_uri=http%3A%2F%2F127.0.0.1%2Fcallback"),
                        "{body_text}"
                    );
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "access_token": "oauth-authcode-token",
                                "refresh_token": "oauth-authcode-refresh",
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
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let protected_metadata_hits = Arc::clone(&protected_metadata_hits);
            let mcp_hits = Arc::clone(&mcp_hits);
            let auth_addr = auth_addr.clone();
            move |req: Request<Incoming>| {
                let protected_metadata_hits = Arc::clone(&protected_metadata_hits);
                let mcp_hits = Arc::clone(&mcp_hits);
                let auth_addr = auth_addr.clone();
                async move {
                    let path = req.uri().path().to_string();
                    if req.method() == Method::GET
                        && path == "/.well-known/oauth-protected-resource/mcp"
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

                    assert_eq!(path, "/mcp");
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
                    assert_eq!(
                        req.headers()
                            .get("authorization")
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer oauth-authcode-token")
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
                            .header("mcp-session-id", "session-oauth-authcode")
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
                                Some("session-oauth-authcode")
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
                        Some("tools/call") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "content": [{
                                            "type": "text",
                                            "text": "Remote MCP result: OAuth auth code worked"
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

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider json");
                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-oauth-authcode-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_oauth_authcode_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp oauth authcode\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("OAuth auth code worked"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-oauth-authcode-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "MCP OAuth authorization code search worked."
                                        },
                                        "finish_reason": "stop"
                                    }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_oauth_authorization_code_mcp_server(
                "docs",
                &format!("http://{mcp_addr}/mcp"),
                "tool-runtime",
                "http://127.0.0.1/callback",
            ),
            &providers,
        )
        .await;
        let flow = api
            .begin_tool_runtime_mcp_oauth_authorization("docs")
            .await
            .expect("tool runtime enabled")
            .expect("start auth flow");
        assert!(flow
            .authorization_url
            .starts_with("https://auth.example.test/authorize?"));
        assert!(flow.authorization_url.contains("response_type=code"));
        assert!(flow.authorization_url.contains("code_challenge="));
        assert!(flow
            .authorization_url
            .contains("code_challenge_method=S256"));
        assert!(flow.authorization_url.contains("state="));
        assert!(flow.authorization_url.contains("resource=http%3A%2F%2F"));
        api.complete_tool_runtime_mcp_oauth_authorization("docs", &flow.state, "auth-code-1")
            .await
            .expect("tool runtime enabled")
            .expect("complete auth flow");

        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the MCP paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("MCP OAuth authorization code search worked.")
        );
        assert_eq!(protected_metadata_hits.load(Ordering::Relaxed), 1);
        assert_eq!(auth_metadata_hits.load(Ordering::Relaxed), 1);
        assert_eq!(token_hits.load(Ordering::Relaxed), 1);
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 4);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn openai_request_uses_sse_mcp_tool() {
        let project_id = "project-openai-mcp-sse";
        let method_log = Arc::new(Mutex::new(Vec::new()));
        let sse_addr = start_mcp_sse_server(Arc::clone(&method_log)).await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-sse-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_sse_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp sse\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Remote MCP result: rust mcp sse"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-sse-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "MCP sse search found the Rust MCP paper."
                                        },
                                        "finish_reason": "stop"
                                    }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_sse_mcp_server("docs", &format!("http://{sse_addr}/sse")),
            &providers,
        )
        .await;
        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the MCP paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("MCP sse search found the Rust MCP paper.")
        );
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);

        let methods = method_log.lock().await.clone();
        assert_eq!(
            methods,
            vec![
                "initialize",
                "notifications/initialized",
                "tools/list",
                "initialize",
                "notifications/initialized",
                "tools/call",
            ]
        );
    }

    #[tokio::test]
    async fn openai_request_fails_closed_when_operator_disables_mcp_server() {
        let project_id = "project-openai-mcp-disabled";
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
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
                            .header("mcp-session-id", "session-disabled-1")
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
                                Some("session-disabled-1")
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
                        Some("tools/call") => {
                            panic!("disabled MCP server should not receive tools/call")
                        }
                        other => panic!("unexpected MCP method: {:?}", other),
                    }
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-disabled-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_disabled_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 11,
                                        "completion_tokens": 5,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        let tool_content = messages[2]["content"].as_str().unwrap_or_default();
                        assert!(tool_content.contains("tool_execution_failed"));
                        assert!(tool_content.contains("disabled by operator"));
                        assert!(tool_content.contains("planned maintenance"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-disabled-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "The MCP server is disabled for maintenance."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 18,
                                        "completion_tokens": 6,
                                        "total_tokens": 24
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let mcp_url = format!("http://{}", mcp_addr);
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_mcp_servers(&[("docs", &mcp_url)]),
            &providers,
        )
        .await;
        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        api.disable_tool_runtime_mcp_server(
            "docs",
            Some("operator-a".to_string()),
            Some("planned maintenance".to_string()),
        )
        .await
        .expect("tool runtime enabled")
        .expect("disable mcp server");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the MCP paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("The MCP server is disabled for maintenance.")
        );
        assert_eq!(
            mcp_hits.load(Ordering::Relaxed),
            3,
            "startup initialize/initialized/tools/list should be the only MCP traffic"
        );
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let runtime_status = api.tool_runtime_status().expect("tool runtime status");
        let docs_server = runtime_status
            .mcp_servers
            .iter()
            .find(|server| server.name == "docs")
            .expect("docs mcp server");
        assert_eq!(docs_server.operator_state, "disabled");
        assert_eq!(
            docs_server.operator_state_reason.as_deref(),
            Some("planned maintenance")
        );
        assert_eq!(docs_server.health_state, "disabled");
        assert_eq!(docs_server.recommended_action.as_deref(), Some("enable"));
        assert_eq!(docs_server.last_call_status.as_deref(), Some("disabled"));
        assert!(docs_server
            .last_call_error
            .as_deref()
            .unwrap_or_default()
            .contains("planned maintenance"));
    }

    #[tokio::test]
    async fn openai_request_retries_retryable_mcp_tool_failure() {
        let project_id = "project-openai-mcp-retry";
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let tool_call_attempts = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            let tool_call_attempts = Arc::clone(&tool_call_attempts);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                let tool_call_attempts = Arc::clone(&tool_call_attempts);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
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
                            .header("mcp-session-id", "session-retry-1")
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
                        Some("notifications/initialized") => Response::builder()
                            .status(StatusCode::ACCEPTED)
                            .body(Full::new(Bytes::new()))
                            .unwrap(),
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
                        Some("tools/call") => {
                            assert_eq!(
                                headers
                                    .get("mcp-session-id")
                                    .and_then(|value| value.to_str().ok()),
                                Some("session-retry-1")
                            );
                            let attempt = tool_call_attempts.fetch_add(1, Ordering::Relaxed);
                            if attempt == 0 {
                                Response::builder()
                                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                                    .body(Full::new(Bytes::from("transient failure")))
                                    .unwrap()
                            } else {
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(
                                        serde_json::json!({
                                            "jsonrpc": "2.0",
                                            "id": body_json["id"].clone(),
                                            "result": {
                                                "content": [{
                                                    "type": "text",
                                                    "text": "Retried MCP result"
                                                }]
                                            }
                                        })
                                        .to_string(),
                                    )))
                                    .unwrap()
                            }
                        }
                        other => panic!("unexpected MCP method: {:?}", other),
                    }
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-retry-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_retry_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 10,
                                        "completion_tokens": 4,
                                        "total_tokens": 14
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Retried MCP result"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-retry-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Recovered from transient MCP failure."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 15,
                                        "completion_tokens": 5,
                                        "total_tokens": 20
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let mcp_url = format!("http://{}", mcp_addr);
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_mcp_server_options(&[(
                "docs",
                &mcp_url,
                Some(1_750),
                Some(1),
            )]),
            &providers,
        )
        .await;
        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the MCP paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Recovered from transient MCP failure.")
        );
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 5);
        assert_eq!(tool_call_attempts.load(Ordering::Relaxed), 2);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let runtime_status = api.tool_runtime_status().expect("tool runtime status");
        let docs_server = runtime_status
            .mcp_servers
            .iter()
            .find(|server| server.name == "docs")
            .expect("docs mcp server");
        assert_eq!(docs_server.total_calls, 1);
        assert_eq!(docs_server.successful_calls, 1);
        assert_eq!(docs_server.failed_calls, 0);
        assert_eq!(docs_server.retried_calls, 1);
        assert_eq!(docs_server.session_reinitializations, 0);
        assert_eq!(docs_server.discovery_refreshes, 1);
        assert_eq!(docs_server.last_discovery_status.as_deref(), Some("ok"));
        assert!(docs_server.last_discovery_at.is_some());
        assert!(docs_server.last_recovery_error.is_none());
        assert_eq!(docs_server.last_call_tool.as_deref(), Some("mcp_search"));
        assert_eq!(docs_server.last_call_status.as_deref(), Some("ok"));
        assert!(docs_server.last_call_at.is_some());
        assert!(docs_server.last_call_error.is_none());
        assert!(docs_server.last_call_http_status.is_none());
    }

    #[tokio::test]
    async fn openai_request_enforces_mcp_tool_call_budget() {
        let project_id = "project-openai-mcp-budget";
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
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
                            .header("mcp-session-id", "session-budget-1")
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
                                Some("session-budget-1")
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
                        Some("tools/call") => {
                            assert_eq!(body_json["params"]["name"].as_str(), Some("search_docs"));
                            Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "jsonrpc": "2.0",
                                        "id": body_json["id"].clone(),
                                        "result": {
                                            "content": [{
                                                "type": "text",
                                                "text": "Remote MCP result: Rust MCP paper"
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

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-budget-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [
                                                {
                                                    "id": "call_mcp_budget_1",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "mcp_search",
                                                        "arguments": "{\"query\":\"rust mcp\"}"
                                                    }
                                                },
                                                {
                                                    "id": "call_mcp_budget_2",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "mcp_search",
                                                        "arguments": "{\"query\":\"another query\"}"
                                                    }
                                                }
                                            ]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 11,
                                        "completion_tokens": 5,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Remote MCP result: Rust MCP paper"));
                        assert_eq!(messages[3]["role"].as_str(), Some("tool"));
                        assert!(messages[3]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("tool_execution_failed"));
                        assert!(messages[3]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("max_calls_per_request 1"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-budget-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Budget limited the second MCP tool call."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 19,
                                        "completion_tokens": 7,
                                        "total_tokens": 26
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let mcp_url = format!("http://{}", mcp_addr);
        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_mcp_server_budget_options(&[(
                "docs",
                &mcp_url,
                None,
                None,
                Some(3),
            )]),
            &providers,
        )
        .await;

        api.upsert_project_tool(mcp_tool_record_with_budget(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
            1,
        ))
        .await
        .expect("governance enabled")
        .expect("upsert tool");

        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{
                        "role": "user",
                        "content": "search twice"
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Budget limited the second MCP tool call.")
        );
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 4);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let runtime_status = api.tool_runtime_status().expect("tool runtime status");
        let docs_server = runtime_status
            .mcp_servers
            .iter()
            .find(|server| server.name == "docs")
            .expect("docs mcp server");
        assert_eq!(docs_server.max_calls_per_request, Some(3));
        assert_eq!(docs_server.total_calls, 2);
        assert_eq!(docs_server.successful_calls, 1);
        assert_eq!(docs_server.failed_calls, 1);
        assert_eq!(docs_server.budget_exceeded_calls, 1);
        assert!(docs_server.last_budget_exceeded_at.is_some());
        assert!(docs_server.last_budget_exceeded_error.is_some());
        assert_eq!(
            docs_server.last_call_status.as_deref(),
            Some("budget_exceeded")
        );
        assert_eq!(docs_server.last_call_tool.as_deref(), Some("mcp_search"));
        assert!(docs_server.last_call_error.is_some());
    }

    #[tokio::test]
    async fn openai_request_enforces_mcp_server_time_budget() {
        let project_id = "project-openai-mcp-time-budget";
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
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
                            .header("mcp-session-id", "session-time-budget-1")
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
                                Some("session-time-budget-1")
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
                        Some("tools/call") => {
                            tokio::time::sleep(Duration::from_millis(75)).await;
                            Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "jsonrpc": "2.0",
                                        "id": body_json["id"].clone(),
                                        "result": {
                                            "content": [{
                                                "type": "text",
                                                "text": "Remote MCP result: slow response"
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

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-time-budget-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [
                                                {
                                                    "id": "call_mcp_time_budget_1",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "mcp_search",
                                                        "arguments": "{\"query\":\"slow rust mcp\"}"
                                                    }
                                                },
                                                {
                                                    "id": "call_mcp_time_budget_2",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "mcp_search",
                                                        "arguments": "{\"query\":\"second slow query\"}"
                                                    }
                                                }
                                            ]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 11,
                                        "completion_tokens": 5,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("tool_execution_failed"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("timed out"));
                        assert_eq!(messages[3]["role"].as_str(), Some("tool"));
                        assert!(messages[3]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("tool_execution_failed"));
                        assert!(messages[3]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("max_total_time_ms 25"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-time-budget-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Time budget limited the MCP server."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 19,
                                        "completion_tokens": 7,
                                        "total_tokens": 26
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let mcp_url = format!("http://{}", mcp_addr);
        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_mcp_server_time_budget_options(&[(
                "docs",
                &mcp_url,
                None,
                None,
                None,
                Some(25),
            )]),
            &providers,
        )
        .await;

        api.upsert_project_tool(mcp_tool_record_with_time_budget(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
            25,
        ))
        .await
        .expect("governance enabled")
        .expect("upsert tool");

        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{
                        "role": "user",
                        "content": "search twice, but enforce a time budget"
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Time budget limited the MCP server.")
        );
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 4);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let runtime_status = api.tool_runtime_status().expect("tool runtime status");
        let docs_server = runtime_status
            .mcp_servers
            .iter()
            .find(|server| server.name == "docs")
            .expect("docs mcp server");
        assert_eq!(docs_server.max_total_time_ms, Some(25));
        assert_eq!(docs_server.total_calls, 2);
        assert_eq!(docs_server.successful_calls, 0);
        assert_eq!(docs_server.failed_calls, 2);
        assert_eq!(docs_server.budget_exceeded_calls, 1);
        assert_eq!(
            docs_server.last_call_status.as_deref(),
            Some("budget_exceeded")
        );
        assert!(docs_server.last_call_error.is_some());
    }

    #[tokio::test]
    async fn openai_request_enforces_mcp_server_output_budget() {
        let project_id = "project-openai-mcp-output-budget";
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
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
                            .header("mcp-session-id", "session-output-budget-1")
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
                                Some("session-output-budget-1")
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
                        Some("tools/call") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "content": [{
                                            "type": "text",
                                            "text": "Remote MCP result: Rust MCP paper"
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

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-output-budget-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [
                                                {
                                                    "id": "call_mcp_output_budget_1",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "mcp_search",
                                                        "arguments": "{\"query\":\"rust mcp\"}"
                                                    }
                                                },
                                                {
                                                    "id": "call_mcp_output_budget_2",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "mcp_search",
                                                        "arguments": "{\"query\":\"another rust mcp\"}"
                                                    }
                                                }
                                            ]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 11,
                                        "completion_tokens": 5,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Remote MCP result: Rust MCP paper"));
                        assert_eq!(messages[3]["role"].as_str(), Some("tool"));
                        let tool_error = messages[3]["content"].as_str().unwrap_or_default();
                        assert!(tool_error.contains("tool_execution_failed"));
                        assert!(tool_error.contains("max_output_tokens 8"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-output-budget-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Output budget limited the second MCP result."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 19,
                                        "completion_tokens": 7,
                                        "total_tokens": 26
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let mcp_url = format!("http://{}", mcp_addr);
        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_mcp_server_output_budget_options(&[(
                "docs",
                &mcp_url,
                None,
                None,
                Some(8),
            )]),
            &providers,
        )
        .await;

        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert tool");

        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{
                        "role": "user",
                        "content": "search twice, but enforce an output budget"
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Output budget limited the second MCP result.")
        );
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 5);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let runtime_status = api.tool_runtime_status().expect("tool runtime status");
        let docs_server = runtime_status
            .mcp_servers
            .iter()
            .find(|server| server.name == "docs")
            .expect("docs mcp server");
        assert_eq!(docs_server.max_output_tokens, Some(8));
        assert_eq!(docs_server.total_calls, 2);
        assert_eq!(docs_server.successful_calls, 1);
        assert_eq!(docs_server.failed_calls, 1);
        assert_eq!(docs_server.budget_exceeded_calls, 1);
        assert_eq!(
            docs_server.last_call_status.as_deref(),
            Some("budget_exceeded")
        );
        assert!(docs_server
            .last_budget_exceeded_error
            .as_deref()
            .unwrap_or_default()
            .contains("max_output_tokens 8"));
    }

    #[tokio::test]
    async fn openai_request_enforces_mcp_tool_output_budget() {
        let project_id = "project-openai-mcp-tool-output-budget";
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
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
                            .header("mcp-session-id", "session-tool-output-budget-1")
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
                                Some("session-tool-output-budget-1")
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
                        Some("tools/call") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "content": [{
                                            "type": "text",
                                            "text": "Remote MCP result: Rust MCP paper"
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

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-tool-output-budget-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_tool_output_budget_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 11,
                                        "completion_tokens": 5,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        let tool_error = messages[2]["content"].as_str().unwrap_or_default();
                        assert!(tool_error.contains("tool_execution_failed"));
                        assert!(
                            tool_error.contains("tool 'mcp_search' exceeded max_output_tokens 4")
                        );
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-tool-output-budget-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Tool output budget blocked the MCP result."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 18,
                                        "completion_tokens": 6,
                                        "total_tokens": 24
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let mcp_url = format!("http://{}", mcp_addr);
        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_mcp_servers(&[("docs", &mcp_url)]),
            &providers,
        )
        .await;

        api.upsert_project_tool(mcp_tool_record_with_output_budget(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
            4,
        ))
        .await
        .expect("governance enabled")
        .expect("upsert tool");

        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{
                        "role": "user",
                        "content": "search once, but enforce a tool output budget"
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Tool output budget blocked the MCP result.")
        );
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 4);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let runtime_status = api.tool_runtime_status().expect("tool runtime status");
        let docs_server = runtime_status
            .mcp_servers
            .iter()
            .find(|server| server.name == "docs")
            .expect("docs mcp server");
        assert_eq!(docs_server.total_calls, 1);
        assert_eq!(docs_server.successful_calls, 0);
        assert_eq!(docs_server.failed_calls, 1);
        assert_eq!(docs_server.budget_exceeded_calls, 1);
        assert_eq!(
            docs_server.last_call_status.as_deref(),
            Some("budget_exceeded")
        );
        assert!(docs_server
            .last_budget_exceeded_error
            .as_deref()
            .unwrap_or_default()
            .contains("max_output_tokens 4"));
    }

    #[tokio::test]
    async fn openai_request_uses_builtin_web_search_executor() {
        let project_id = "project-openai-web-search";
        let search_hits = Arc::new(AtomicUsize::new(0));
        let search_requests = Arc::new(Mutex::new(Vec::new()));
        let search_addr = start_upstream_async({
            let search_hits = Arc::clone(&search_hits);
            let search_requests = Arc::clone(&search_requests);
            move |req: Request<Incoming>| {
                let search_hits = Arc::clone(&search_hits);
                let search_requests = Arc::clone(&search_requests);
                async move {
                    search_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect search body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("search request json");
                    search_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "results": [{
                                    "title": "Rust RFC 123",
                                    "url": "https://example.com/rfc-123"
                                }]
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| panic!(
                            "provider request json: {error}; body={}",
                            String::from_utf8_lossy(&body)
                        ));

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-tool-search",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_search_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "web_search",
                                                    "arguments": "{\"query\":\"rust rfc\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 10,
                                        "completion_tokens": 4,
                                        "total_tokens": 14
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Rust RFC 123"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-final-search",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Found Rust RFC 123 from the built-in search."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 18,
                                        "completion_tokens": 8,
                                        "total_tokens": 26
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_backends(
                &[("local-search", &format!("http://{}", search_addr))],
                None,
            ),
            &providers,
        )
        .await;
        api.upsert_project_tool(web_search_tool_record(
            project_id,
            "web_search",
            "local-search",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Found Rust RFC 123 from the built-in search.")
        );
        assert_eq!(search_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let search_requests = search_requests.lock().await;
        assert_eq!(search_requests[0]["query"].as_str(), Some("rust rfc"));
    }

    #[tokio::test]
    async fn openai_request_recovers_after_mcp_session_expiry() {
        let project_id = "project-openai-mcp-recover";
        let mcp_hits = Arc::new(AtomicUsize::new(0));
        let session_generation = Arc::new(AtomicUsize::new(0));
        let expired_once = Arc::new(AtomicBool::new(false));
        let mcp_addr = start_upstream_async({
            let mcp_hits = Arc::clone(&mcp_hits);
            let session_generation = Arc::clone(&session_generation);
            let expired_once = Arc::clone(&expired_once);
            move |req: Request<Incoming>| {
                let mcp_hits = Arc::clone(&mcp_hits);
                let session_generation = Arc::clone(&session_generation);
                let expired_once = Arc::clone(&expired_once);
                async move {
                    mcp_hits.fetch_add(1, Ordering::Relaxed);
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
                            let generation = session_generation.fetch_add(1, Ordering::Relaxed) + 1;
                            let session_id = format!("session-recover-{generation}");
                            Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .header("mcp-session-id", session_id)
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
                        Some("notifications/initialized") => Response::builder()
                            .status(StatusCode::ACCEPTED)
                            .body(Full::new(Bytes::new()))
                            .unwrap(),
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
                        Some("tools/call") => {
                            let session_id = headers
                                .get("mcp-session-id")
                                .and_then(|value| value.to_str().ok())
                                .unwrap_or_default()
                                .to_string();
                            if session_id == "session-recover-1"
                                && !expired_once.swap(true, Ordering::Relaxed)
                            {
                                Response::builder()
                                    .status(StatusCode::NOT_FOUND)
                                    .body(Full::new(Bytes::from("session expired")))
                                    .unwrap()
                            } else {
                                assert_eq!(session_id, "session-recover-2");
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(
                                        serde_json::json!({
                                            "jsonrpc": "2.0",
                                            "id": body_json["id"].clone(),
                                            "result": {
                                                "content": [{
                                                    "type": "text",
                                                    "text": "Recovered MCP result"
                                                }]
                                            }
                                        })
                                        .to_string(),
                                    )))
                                    .unwrap()
                            }
                        }
                        other => panic!("unexpected MCP method: {:?}", other),
                    }
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-recover-1",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_mcp_recover_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "mcp_search",
                                                    "arguments": "{\"query\":\"rust mcp\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 12,
                                        "completion_tokens": 5,
                                        "total_tokens": 17
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert_eq!(
                            messages[2]["content"].as_str(),
                            Some("Recovered MCP result")
                        );
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-mcp-recover-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Recovered after MCP session reset."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 18,
                                        "completion_tokens": 6,
                                        "total_tokens": 24
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let mcp_url = format!("http://{}", mcp_addr);
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_mcp_servers(&[("docs", &mcp_url)]),
            &providers,
        )
        .await;
        api.upsert_project_tool(mcp_tool_record(
            project_id,
            "mcp_search",
            "docs",
            "search_docs",
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "recover the MCP search"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["mcp_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Recovered after MCP session reset.")
        );
        assert_eq!(mcp_hits.load(Ordering::Relaxed), 8);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        assert_eq!(session_generation.load(Ordering::Relaxed), 2);
        let runtime_status = api.tool_runtime_status().expect("tool runtime status");
        let docs_server = runtime_status
            .mcp_servers
            .iter()
            .find(|server| server.name == "docs")
            .expect("docs mcp server");
        assert_eq!(docs_server.session_reinitializations, 1);
        assert!(docs_server.last_session_reinitialized_at.is_some());
        assert!(docs_server.last_recovery_error.is_none());
    }

    #[tokio::test]
    async fn anthropic_request_uses_builtin_arxiv_executor() {
        let project_id = "project-anthropic-arxiv";
        let arxiv_hits = Arc::new(AtomicUsize::new(0));
        let arxiv_addr = start_upstream_async({
            let arxiv_hits = Arc::clone(&arxiv_hits);
            move |req: Request<Incoming>| {
                let arxiv_hits = Arc::clone(&arxiv_hits);
                async move {
                    arxiv_hits.fetch_add(1, Ordering::Relaxed);
                    let query = req.uri().query().unwrap_or_default().to_string();
                    assert!(query.contains("search_query=all%3Aattention%20is%20all%20you%20need"));
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/atom+xml")
                        .body(Full::new(Bytes::from(concat!(
                            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                            "<feed xmlns=\"http://www.w3.org/2005/Atom\">",
                            "<entry>",
                            "<id>http://arxiv.org/abs/1706.03762v1</id>",
                            "<title>Attention Is All You Need</title>",
                            "<summary>Transformer paper.</summary>",
                            "<published>2017-06-12T17:57:16Z</published>",
                            "<author><name>Ashish Vaswani</name></author>",
                            "<author><name>Noam Shazeer</name></author>",
                            "</entry>",
                            "</feed>"
                        ))))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| panic!(
                            "provider request json: {error}; body={}",
                            String::from_utf8_lossy(&body)
                        ));

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "msg_arxiv_1",
                                    "type": "message",
                                    "role": "assistant",
                                    "model": "claude-sonnet-4-20250514",
                                    "content": [{
                                        "type": "tool_use",
                                        "id": "toolu_arxiv_1",
                                        "name": "arxiv_search",
                                        "input": {
                                            "query": "attention is all you need"
                                        }
                                    }],
                                    "stop_reason": "tool_use",
                                    "usage": {
                                        "input_tokens": 14,
                                        "output_tokens": 5
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        let tool_content = messages[2]["content"][0]["content"]
                            .as_str()
                            .unwrap_or_default();
                        assert!(tool_content.contains("1706.03762"));
                        assert!(tool_content.contains("Attention Is All You Need"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "msg_arxiv_2",
                                    "type": "message",
                                    "role": "assistant",
                                    "model": "claude-sonnet-4-20250514",
                                    "content": [{
                                        "type": "text",
                                        "text": "The paper is Attention Is All You Need (arXiv:1706.03762)."
                                    }],
                                    "stop_reason": "end_turn",
                                    "usage": {
                                        "input_tokens": 18,
                                        "output_tokens": 7
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![anthropic_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::Anthropic,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_backends(
                &[],
                Some(&format!("http://{}/api/query", arxiv_addr)),
            ),
            &providers,
        )
        .await;
        api.upsert_project_tool(arxiv_tool_record(project_id, "arxiv_search", 1))
            .await
            .expect("governance enabled")
            .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "max_tokens": 256,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": "find the paper"
                        }]
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["arxiv_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["content"][0]["text"].as_str(),
            Some("The paper is Attention Is All You Need (arXiv:1706.03762).")
        );
        assert_eq!(arxiv_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn startup_validation_rejects_unknown_web_search_backend() {
        let project_id = "project-invalid-web-search";
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let store = plugin_llm_gateway::store::connect(&store_url)
            .await
            .expect("connect store");
        store
            .upsert_project_tool(&web_search_tool_record(
                project_id,
                "web_search",
                "missing-backend",
            ))
            .await
            .expect("seed project tool");

        let providers = vec![openai_provider(
            "http://127.0.0.1:9999".to_string(),
            ProviderToolProtocol::OpenAi,
        )];

        let result = plugin_llm_gateway::create_plugins_with_options(
            &tool_runtime_config_with_backends(&[], None),
            Some(&store_url),
            &providers,
            &[],
            CreatePluginsOptions::default(),
            None,
        )
        .await;

        let error = match result {
            Ok(_) => panic!("startup should reject missing backend"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("references unknown web_search backend 'missing-backend'"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn startup_validation_rejects_unknown_mcp_server() {
        let project_id = "project-invalid-mcp";
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let store = plugin_llm_gateway::store::connect(&store_url)
            .await
            .expect("connect store");
        store
            .upsert_project_tool(&mcp_tool_record(
                project_id,
                "mcp_search",
                "missing-mcp",
                "search_docs",
            ))
            .await
            .expect("seed project tool");

        let providers = vec![openai_provider(
            "http://127.0.0.1:9999".to_string(),
            ProviderToolProtocol::OpenAi,
        )];

        let result = plugin_llm_gateway::create_plugins_with_options(
            &tool_runtime_config(),
            Some(&store_url),
            &providers,
            &[],
            CreatePluginsOptions::default(),
            None,
        )
        .await;

        let error = match result {
            Ok(_) => panic!("startup should reject missing mcp server"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("references unknown mcp server 'missing-mcp'"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn startup_validation_rejects_missing_remote_mcp_tool() {
        let project_id = "project-invalid-remote-mcp-tool";
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let store = plugin_llm_gateway::store::connect(&store_url)
            .await
            .expect("connect store");
        store
            .upsert_project_tool(&mcp_tool_record(
                project_id,
                "mcp_search",
                "docs",
                "missing_tool",
            ))
            .await
            .expect("seed project tool");

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
                    .header("mcp-session-id", "session-test-2")
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
                        Some("session-test-2")
                    );
                    Response::builder()
                        .status(StatusCode::ACCEPTED)
                        .body(Full::new(Bytes::new()))
                        .unwrap()
                }
                Some("tools/list") => {
                    assert_eq!(
                        headers
                            .get("mcp-session-id")
                            .and_then(|value| value.to_str().ok()),
                        Some("session-test-2")
                    );
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
        })
        .await;

        let providers = vec![openai_provider(
            "http://127.0.0.1:9999".to_string(),
            ProviderToolProtocol::OpenAi,
        )];
        let mcp_url = format!("http://{}", mcp_addr);

        let result = plugin_llm_gateway::create_plugins_with_options(
            &tool_runtime_config_with_mcp_servers(&[("docs", &mcp_url)]),
            Some(&store_url),
            &providers,
            &[],
            CreatePluginsOptions::default(),
            None,
        )
        .await;

        let error = match result {
            Ok(_) => panic!("startup should reject missing remote MCP tool"),
            Err(error) => error,
        };

        assert!(
            error.to_string().contains(
                "references remote MCP tool 'missing_tool' that was not found on server 'docs'"
            ),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn openai_streaming_request_emits_gateway_tool_events() {
        let project_id = "project-openai-stream";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            move |req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect tool body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("tool request json");
                    assert_eq!(
                        body_json["arguments"]["query"].as_str(),
                        Some("rust streaming")
                    );
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"Rust streaming RFC"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| panic!(
                            "provider request json: {error}; body={}",
                            String::from_utf8_lossy(&body)
                        ));

                    if call_index == 0 {
                        assert!(body_json.get("trp_tools").is_none());
                        assert_eq!(body_json["stream"].as_bool(), Some(true));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "text/event-stream")
                            .body(Full::new(Bytes::from(
                                concat!(
                                    "data: {\"id\":\"chatcmpl-tool-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
                                    "data: {\"id\":\"chatcmpl-tool-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_stream_1\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"rust streaming\\\"}\"}}]}}]}\n\n",
                                    "data: {\"id\":\"chatcmpl-tool-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                                    "data: {\"id\":\"chatcmpl-tool-1\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}\n\n",
                                    "data: [DONE]\n\n"
                                ),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[2]["role"].as_str(), Some("tool"));
                        assert!(messages[2]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Rust streaming RFC"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "text/event-stream")
                            .body(Full::new(Bytes::from(
                                concat!(
                                    "data: {\"id\":\"chatcmpl-final\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
                                    "data: {\"id\":\"chatcmpl-final\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Found the \"}}]}\n\n",
                                    "data: {\"id\":\"chatcmpl-final\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"paper via tool.\"}}]}\n\n",
                                    "data: {\"id\":\"chatcmpl-final\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                                    "data: {\"id\":\"chatcmpl-final\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":18,\"completion_tokens\":7,\"total_tokens\":25}}\n\n",
                                    "data: [DONE]\n\n"
                                ),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "stream": true,
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .contains("text/event-stream"));
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_text = String::from_utf8_lossy(&body);
        assert!(body_text.contains("event: trp_tool_call"), "{body_text}");
        assert!(body_text.contains("event: trp_tool_result"), "{body_text}");
        assert!(
            body_text.contains("\"tool_name\":\"web_search\""),
            "{body_text}"
        );
        assert!(body_text.contains("Found the "), "{body_text}");
        assert!(body_text.contains("paper via tool."), "{body_text}");
        assert_eq!(tool_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn openai_responses_streaming_request_is_rejected_in_strict_mode() {
        let project_id = "project-openai-responses-stream-strict";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            move |_req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"unexpected"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |_req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(Full::new(Bytes::from("data: [DONE]\n\n")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_responses_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
            .uri("/v1/responses")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "stream": true,
                    "input": [{
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": "find the paper"
                        }]
                    }],
                    "tool_choice": {
                        "type": "allowed_tools",
                        "tools": ["web_search"]
                    },
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp
            .collect()
            .await
            .expect("collect error response")
            .to_bytes();
        let body_text = String::from_utf8_lossy(&body);
        assert!(body_text.contains("responses_stream_mode"), "{body_text}");
        assert!(body_text.contains("composed"), "{body_text}");
        assert_eq!(provider_hits.load(Ordering::Relaxed), 0);
        assert_eq!(tool_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn openai_responses_streaming_request_uses_registered_webhook_tool() {
        let project_id = "project-openai-responses-stream";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            move |req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect tool body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("tool request json");
                    assert_eq!(body_json["arguments"]["query"].as_str(), Some("rust rfc"));
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"Rust RFC paper"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    assert_eq!(req.uri().path(), "/v1/responses");
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| panic!(
                            "provider request json: {error}; body={}",
                            String::from_utf8_lossy(&body)
                        ));

                    if call_index == 0 {
                        assert!(body_json.get("trp_tools").is_none());
                        assert_eq!(body_json["stream"].as_bool(), Some(true));
                        assert_eq!(
                            body_json["tool_choice"]["type"].as_str(),
                            Some("allowed_tools")
                        );
                        assert_eq!(
                            body_json["tool_choice"]["tools"]
                                .as_array()
                                .expect("allowed tools")
                                .iter()
                                .filter_map(|value| value.as_str())
                                .collect::<Vec<_>>(),
                            vec!["web_search"]
                        );
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "text/event-stream")
                            .body(Full::new(Bytes::from(
                                concat!(
                                    "event: response.created\n",
                                    "data: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"resp_stream_1\",\"object\":\"response\",\"model\":\"gpt-4o\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
                                    "event: response.output_item.added\n",
                                    "data: {\"type\":\"response.output_item.added\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_stream_1\",\"status\":\"in_progress\",\"call_id\":\"call_stream_1\",\"name\":\"web_search\",\"arguments\":\"\"}}\n\n",
                                    "event: response.output_item.done\n",
                                    "data: {\"type\":\"response.output_item.done\",\"sequence_number\":3,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_stream_1\",\"status\":\"completed\",\"call_id\":\"call_stream_1\",\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"rust rfc\\\"}\"}}\n\n",
                                    "event: response.completed\n",
                                    "data: {\"type\":\"response.completed\",\"sequence_number\":4,\"response\":{\"id\":\"resp_stream_1\",\"object\":\"response\",\"model\":\"gpt-4o\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":12,\"output_tokens\":4}}}\n\n",
                                    "data: [DONE]\n\n"
                                ),
                            )))
                            .unwrap()
                    } else {
                        assert_eq!(
                            body_json["previous_response_id"].as_str(),
                            Some("resp_stream_1")
                        );
                        let input = body_json["input"].as_array().expect("input array");
                        assert_eq!(input.len(), 1);
                        assert_eq!(
                            input[0]["type"].as_str(),
                            Some("function_call_output")
                        );
                        assert_eq!(input[0]["call_id"].as_str(), Some("call_stream_1"));
                        assert!(input[0]["output"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("Rust RFC paper"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "text/event-stream")
                            .body(Full::new(Bytes::from(
                                concat!(
                                    "event: response.created\n",
                                    "data: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"resp_stream_2\",\"object\":\"response\",\"model\":\"gpt-4o\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
                                    "event: response.output_item.added\n",
                                    "data: {\"type\":\"response.output_item.added\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_stream_1\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}\n\n",
                                    "event: response.content_part.added\n",
                                    "data: {\"type\":\"response.content_part.added\",\"sequence_number\":3,\"item_id\":\"msg_stream_1\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"annotations\":[],\"text\":\"\"}}\n\n",
                                    "event: response.output_text.delta\n",
                                    "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":4,\"item_id\":\"msg_stream_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Found the \"}\n\n",
                                    "event: response.output_text.delta\n",
                                    "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":5,\"item_id\":\"msg_stream_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Rust RFC paper via the tool.\"}\n\n",
                                    "event: response.output_text.done\n",
                                    "data: {\"type\":\"response.output_text.done\",\"sequence_number\":6,\"item_id\":\"msg_stream_1\",\"output_index\":0,\"content_index\":0,\"text\":\"Found the Rust RFC paper via the tool.\"}\n\n",
                                    "event: response.content_part.done\n",
                                    "data: {\"type\":\"response.content_part.done\",\"sequence_number\":7,\"item_id\":\"msg_stream_1\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"annotations\":[],\"text\":\"Found the Rust RFC paper via the tool.\"}}\n\n",
                                    "event: response.output_item.done\n",
                                    "data: {\"type\":\"response.output_item.done\",\"sequence_number\":8,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_stream_1\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Found the Rust RFC paper via the tool.\"}]}}\n\n",
                                    "event: response.completed\n",
                                    "data: {\"type\":\"response.completed\",\"sequence_number\":9,\"response\":{\"id\":\"resp_stream_2\",\"object\":\"response\",\"model\":\"gpt-4o\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":18,\"output_tokens\":7}}}\n\n",
                                    "data: [DONE]\n\n"
                                ),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_responses_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            tool_runtime_config_with_responses_stream_mode("composed"),
            &providers,
        )
        .await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
            .uri("/v1/responses")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "stream": true,
                    "input": [{
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": "find the paper"
                        }]
                    }],
                    "tool_choice": {
                        "type": "allowed_tools",
                        "tools": ["web_search"]
                    },
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .contains("text/event-stream"));
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_text = String::from_utf8_lossy(&body);
        assert!(!body_text.contains("event: trp_tool_call"), "{body_text}");
        assert!(!body_text.contains("event: trp_tool_result"), "{body_text}");
        assert!(
            body_text.contains("\"type\":\"function_call_output\""),
            "{body_text}"
        );
        assert!(body_text.contains("\"sequence_number\":1"), "{body_text}");
        assert!(body_text.contains("\"sequence_number\":6"), "{body_text}");
        assert!(
            body_text.contains("Found the Rust RFC paper via the tool."),
            "{body_text}"
        );
        assert!(
            body_text.contains("response.output_text.delta"),
            "{body_text}"
        );
        assert_eq!(tool_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn openai_responses_request_rejects_disallowed_tool_call() {
        let project_id = "project-openai-responses-disallowed";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            move |_req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"unexpected"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    assert_eq!(call_index, 0, "gateway should reject before retrying");
                    assert_eq!(req.uri().path(), "/v1/responses");
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| {
                            panic!(
                                "provider request json: {error}; body={}",
                                String::from_utf8_lossy(&body)
                            )
                        });
                    assert_eq!(
                        body_json["tool_choice"]["type"].as_str(),
                        Some("allowed_tools")
                    );
                    assert_eq!(
                        body_json["tool_choice"]["tools"]
                            .as_array()
                            .expect("allowed tools")
                            .iter()
                            .filter_map(|value| value.as_str())
                            .collect::<Vec<_>>(),
                        vec!["web_search"]
                    );
                    let tool_names = body_json["tools"]
                        .as_array()
                        .expect("tools array")
                        .iter()
                        .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
                        .collect::<Vec<_>>();
                    assert!(tool_names.contains(&"web_search"));
                    assert!(tool_names.contains(&"arxiv_search"));
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "resp_tool_disallowed_1",
                                "object": "response",
                                "model": "gpt-4o",
                                "status": "completed",
                                "output": [{
                                    "type": "function_call",
                                    "id": "fc_disallowed_1",
                                    "call_id": "call_disallowed_1",
                                    "name": "arxiv_search",
                                    "arguments": "{\"query\":\"rust rfc\"}",
                                    "status": "completed"
                                }],
                                "usage": {
                                    "input_tokens": 12,
                                    "output_tokens": 4,
                                    "total_tokens": 16
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_responses_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert web_search");
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "arxiv_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert arxiv_search");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
            .uri("/v1/responses")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "input": [{
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": "find the paper"
                        }]
                    }],
                    "tool_choice": {
                        "type": "allowed_tools",
                        "tools": ["web_search"]
                    },
                    "trp_tools": {
                        "enabled": true
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let body = resp
            .collect()
            .await
            .expect("collect error response")
            .to_bytes();
        let body_text = String::from_utf8_lossy(&body);
        assert!(body_text.contains("disallowed tool"), "{body_text}");
        assert!(body_text.contains("arxiv_search"), "{body_text}");
        assert_eq!(tool_hits.load(Ordering::Relaxed), 0);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn anthropic_streaming_request_emits_gateway_tool_events() {
        let project_id = "project-anthropic-stream";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            move |req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect tool body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("tool request json");
                    assert_eq!(
                        body_json["arguments"]["query"].as_str(),
                        Some("attention is all you need")
                    );
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"arXiv:1706.03762"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value = serde_json::from_slice(&body)
                        .unwrap_or_else(|error| panic!(
                            "provider request json: {error}; body={}",
                            String::from_utf8_lossy(&body)
                        ));

                    if call_index == 0 {
                        assert!(body_json.get("trp_tools").is_none());
                        assert_eq!(body_json["stream"].as_bool(), Some(true));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "text/event-stream")
                            .body(Full::new(Bytes::from(
                                concat!(
                                    "event: message_start\n",
                                    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-20250514\",\"content\":[]}}\n\n",
                                    "event: content_block_start\n",
                                    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_stream_1\",\"name\":\"arxiv_search\",\"input\":{\"query\":\"attention is all you need\"}}}\n\n",
                                    "event: content_block_stop\n",
                                    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                                    "event: message_delta\n",
                                    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"input_tokens\":15,\"output_tokens\":5}}\n\n",
                                    "event: message_stop\n",
                                    "data: {\"type\":\"message_stop\"}\n\n"
                                ),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[2]["role"].as_str(), Some("user"));
                        assert!(messages[2]["content"][0]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("1706.03762"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "text/event-stream")
                            .body(Full::new(Bytes::from(
                                concat!(
                                    "event: message_start\n",
                                    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-20250514\",\"content\":[]}}\n\n",
                                    "event: content_block_start\n",
                                    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                                    "event: content_block_delta\n",
                                    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"The paper is \"}}\n\n",
                                    "event: content_block_delta\n",
                                    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"arXiv:1706.03762.\"}}\n\n",
                                    "event: content_block_stop\n",
                                    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                                    "event: message_delta\n",
                                    "data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":19,\"output_tokens\":8}}\n\n",
                                    "event: message_stop\n",
                                    "data: {\"type\":\"message_stop\"}\n\n"
                                ),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![anthropic_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::Anthropic,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "arxiv_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "stream": true,
                    "max_tokens": 256,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": "find the paper"
                        }]
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["arxiv_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .contains("text/event-stream"));
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_text = String::from_utf8_lossy(&body);
        assert!(body_text.contains("event: trp_tool_call"), "{body_text}");
        assert!(body_text.contains("event: trp_tool_result"), "{body_text}");
        assert!(
            body_text.contains("\"tool_name\":\"arxiv_search\""),
            "{body_text}"
        );
        assert!(body_text.contains("The paper is "), "{body_text}");
        assert!(body_text.contains("arXiv:1706.03762."), "{body_text}");
        assert_eq!(tool_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn openai_tool_output_is_redacted_and_audited() {
        let project_id = "project-openai-audit";
        let request_secret = "ghp_abcdefghijklmnopqrstuvwxyz12345";
        let tool_secret = "ghp_zyxwvutsrqponmlkjihgfedcba12345";
        let (endpoint, _dir) = start_semantic_service().await;

        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            move |_req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "result": format!("Company X layoffs next week {}", tool_secret),
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_requests = Arc::new(Mutex::new(Vec::new()));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            let provider_requests = Arc::clone(&provider_requests);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                let provider_requests = Arc::clone(&provider_requests);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-tool-redact",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_redact",
                                                "type": "function",
                                                "function": {
                                                    "name": "web_search",
                                                    "arguments": "{\"query\":\"company x layoffs\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 12,
                                        "completion_tokens": 4,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let body = req
                            .into_body()
                            .collect()
                            .await
                            .expect("collect provider body")
                            .to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body)
                            .unwrap_or_else(|error| {
                                panic!(
                                    "provider request json: {error}; body={}",
                                    String::from_utf8_lossy(&body)
                                )
                            });
                        provider_requests.lock().await.push(body_json);
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-final-redact",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Redaction and semantic audit completed."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 18,
                                        "completion_tokens": 7,
                                        "total_tokens": 25
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            plugin_configs(vec![
                content_filter_config("redact_and_forward"),
                semantic_safety_config(&endpoint),
                cost_tracker_config(),
            ]),
            &providers,
        )
        .await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        tokio::time::timeout(
            Duration::from_secs(5),
            api.upsert_semantic_policy(semantic_policy_record(project_id)),
        )
        .await
        .expect("semantic policy upsert timed out")
        .expect("semantic enabled")
        .expect("upsert semantic policy");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{
                        "role": "user",
                        "content": format!("Store this secret {} for later", request_secret),
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = match tokio::time::timeout(
            Duration::from_secs(5),
            send_request(&proxy_addr, req),
        )
        .await
        {
            Ok(resp) => resp,
            Err(error) => panic!(
                "proxy request timed out: {error:?}; provider_hits={}, tool_hits={}",
                provider_hits.load(Ordering::Relaxed),
                tool_hits.load(Ordering::Relaxed)
            ),
        };
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Redaction and semantic audit completed.")
        );
        assert_eq!(tool_hits.load(Ordering::Relaxed), 1);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        let provider_requests = provider_requests.lock().await;
        assert_eq!(provider_requests.len(), 1);
        let second_messages = provider_requests[0]["messages"]
            .as_array()
            .expect("messages");
        assert!(!second_messages[0]["content"]
            .as_str()
            .unwrap_or_default()
            .contains(request_secret));
        let tool_content = second_messages[2]["content"].as_str().unwrap_or_default();
        assert!(!tool_content.contains(tool_secret));
        assert!(tool_content.contains("Company X layoffs next week"));

        let logs = wait_for_request_logs(&api, project_id).await;
        assert!(!logs.is_empty(), "expected request log entry");
        let log = &logs[0];
        assert_eq!(log.safety_mode.as_deref(), Some("redact_and_forward"));
        assert_eq!(log.semantic_policy_version.as_deref(), Some("v1"));
        assert!(
            log.safety_matches
                .as_deref()
                .unwrap_or_default()
                .contains("github"),
            "expected safety matches in {:?}",
            log.safety_matches
        );
        assert!(
            log.semantic_findings
                .as_deref()
                .unwrap_or_default()
                .contains("company-x"),
            "expected semantic findings in {:?}",
            log.semantic_findings
        );
    }

    #[tokio::test]
    async fn blocked_tool_output_becomes_structured_error() {
        let project_id = "project-openai-block";
        let tool_secret = "ghp_blockedtooloutputabcdefghijklmnopqrstuvwxyz";

        let tool_addr = start_upstream_async(move |_req: Request<Incoming>| async move {
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "result": format!("secret {}", tool_secret),
                    })
                    .to_string(),
                )))
                .unwrap()
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-tool-block",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": serde_json::Value::Null,
                                            "tool_calls": [{
                                                "id": "call_block",
                                                "type": "function",
                                                "function": {
                                                    "name": "web_search",
                                                    "arguments": "{\"query\":\"blocked\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 12,
                                        "completion_tokens": 4,
                                        "total_tokens": 16
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let messages = body_json["messages"].as_array().expect("messages");
                        let tool_content = messages[2]["content"].as_str().unwrap_or_default();
                        assert!(tool_content.contains("tool_output_blocked"));
                        assert!(!tool_content.contains(tool_secret));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-final-block",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Tool output was blocked."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 18,
                                        "completion_tokens": 7,
                                        "total_tokens": 25
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway_with_configs(
            plugin_configs(vec![content_filter_config("block")]),
            &providers,
        )
        .await;
        api.upsert_safety_policy(SafetyPolicyRecord {
            project_id: project_id.to_string(),
            mode: "block".to_string(),
            rules_json: None,
            updated_at: "0".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("upsert safety policy");
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find blocked data"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Tool output was blocked.")
        );
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn anthropic_tool_failure_is_returned_in_band() {
        let project_id = "project-anthropic-error";

        let tool_addr = start_upstream_async(move |_req: Request<Incoming>| async move {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(r#"{"error":"boom"}"#)))
                .unwrap()
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");

                    if call_index == 0 {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "msg_err_1",
                                    "type": "message",
                                    "role": "assistant",
                                    "model": "claude-sonnet-4-20250514",
                                    "content": [{
                                        "type": "tool_use",
                                        "id": "toolu_err_1",
                                        "name": "arxiv_search",
                                        "input": {
                                            "query": "attention is all you need"
                                        }
                                    }],
                                    "stop_reason": "tool_use",
                                    "usage": {
                                        "input_tokens": 15,
                                        "output_tokens": 5
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        let content = body_json["messages"][2]["content"]
                            .as_array()
                            .expect("tool result content");
                        let tool_content = content[0]["content"].as_str().unwrap_or_default();
                        assert!(tool_content.contains("tool_execution_failed"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "msg_err_2",
                                    "type": "message",
                                    "role": "assistant",
                                    "model": "claude-sonnet-4-20250514",
                                    "content": [{
                                        "type": "text",
                                        "text": "Recovered from tool failure."
                                    }],
                                    "stop_reason": "end_turn",
                                    "usage": {
                                        "input_tokens": 19,
                                        "output_tokens": 8
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![anthropic_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::Anthropic,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "arxiv_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "max_tokens": 256,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": "find the paper"
                        }]
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["arxiv_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["content"][0]["text"].as_str(),
            Some("Recovered from tool failure.")
        );
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn provider_without_tool_protocol_rejects_trp_tools() {
        let project_id = "project-no-protocol";
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |_req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("ok")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::None,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", provider_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp
            .collect()
            .await
            .expect("collect error response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("error json");
        assert_eq!(
            body_json["error"]["message"].as_str(),
            Some("provider does not support gateway-managed tools")
        );
        assert_eq!(provider_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn disabled_project_tool_is_rejected() {
        let project_id = "project-disabled-tool";
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |_req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("ok")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        let mut tool = webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", provider_addr),
        );
        tool.enabled = false;
        api.upsert_project_tool(tool)
            .await
            .expect("governance enabled")
            .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp
            .collect()
            .await
            .expect("collect error response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("error json");
        assert!(body_json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("tool 'web_search' is not enabled for this project"));
        assert_eq!(provider_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn project_tool_allow_list_filters_default_tool_selection() {
        let project_id = "project-tool-allow-list";
        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_addr = start_upstream_async({
            let tool_hits = Arc::clone(&tool_hits);
            move |_req: Request<Incoming>| {
                let tool_hits = Arc::clone(&tool_hits);
                async move {
                    tool_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"result":"approved tool"}"#)))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    let call_index = provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");
                    if call_index == 0 {
                        let tool_names = body_json["tools"]
                            .as_array()
                            .expect("tools")
                            .iter()
                            .filter_map(|tool| {
                                tool.get("function")
                                    .and_then(|value| value.get("name"))
                                    .and_then(|value| value.as_str())
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(tool_names, vec!["web_search"]);
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-tool-allow-list",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": null,
                                            "tool_calls": [{
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "web_search",
                                                    "arguments": "{\"query\":\"rust book\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 10,
                                        "completion_tokens": 4,
                                        "total_tokens": 14
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    } else {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "chatcmpl-tool-allow-list-final",
                                    "object": "chat.completion",
                                    "model": "gpt-4o",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "Only the approved tool was exposed."
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 12,
                                        "completion_tokens": 5,
                                        "total_tokens": 17
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "arxiv_search",
            &format!("http://{}", tool_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        api.upsert_project_policy(plugin_llm_gateway::store::ProjectPolicyRecord {
            project_id: project_id.to_string(),
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
            semantic_cache_enabled: None,
            semantic_cache_ttl_secs: None,
            semantic_cache_similarity_threshold: None,
            tool_approval_mode: Some("allow_list".to_string()),
            allowed_tools: Some(r#"["web_search"]"#.to_string()),
            updated_at: "0".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("upsert project policy");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "trp_tools": {
                        "enabled": true
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp
            .collect()
            .await
            .expect("collect final response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("final json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Only the approved tool was exposed.")
        );
        assert_eq!(provider_hits.load(Ordering::Relaxed), 2);
        assert_eq!(tool_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn virtual_key_tool_policy_can_deny_all_managed_tools() {
        let project_id = "project-tool-deny-all";
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |_req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("ok")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", provider_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key_with_runtime_policy(
                Some(project_id),
                "tool-key",
                "openai",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("deny_all".to_string()),
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp
            .collect()
            .await
            .expect("collect error response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("error json");
        assert_eq!(
            body_json["error"]["message"].as_str(),
            Some("managed tools are disabled by runtime policy")
        );
        assert_eq!(provider_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn virtual_key_tool_allow_list_rejects_unapproved_tool() {
        let project_id = "project-tool-key-allow-list";
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |_req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("ok")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", provider_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "arxiv_search",
            &format!("http://{}", provider_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key_with_runtime_policy(
                Some(project_id),
                "tool-key",
                "openai",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(vec!["web_search".to_string()]),
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["arxiv_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp
            .collect()
            .await
            .expect("collect error response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("error json");
        assert!(body_json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("tool 'arxiv_search' is not approved by runtime policy"));
        assert_eq!(provider_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn request_merges_project_and_client_tools() {
        let project_id = "project-tool-merge";
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");
                    let tool_names = body_json["tools"]
                        .as_array()
                        .expect("tools")
                        .iter()
                        .filter_map(|tool| {
                            tool.get("function")
                                .and_then(|value| value.get("name"))
                                .and_then(|value| value.as_str())
                        })
                        .collect::<Vec<_>>();
                    assert!(tool_names.contains(&"web_search"));
                    assert!(tool_names.contains(&"client_tool"));
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-merge",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Both tool definitions arrived."
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 4,
                                    "total_tokens": 14
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", provider_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "client_tool",
                            "description": "Client tool",
                            "parameters": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(provider_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn request_rejects_tool_name_collision() {
        let project_id = "project-tool-collision";
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |_req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("ok")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_provider(
            format!("http://{}", provider_addr),
            ProviderToolProtocol::OpenAi,
        )];
        let (plugins, api) = setup_gateway(&providers).await;
        api.upsert_project_tool(webhook_tool_record(
            project_id,
            "web_search",
            &format!("http://{}", provider_addr),
        ))
        .await
        .expect("governance enabled")
        .expect("upsert project tool");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "tool-key",
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
                    "messages": [{"role": "user", "content": "find the paper"}],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "Client tool",
                            "parameters": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    }],
                    "trp_tools": {
                        "enabled": true,
                        "names": ["web_search"]
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp
            .collect()
            .await
            .expect("collect error response")
            .to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("error json");
        assert!(body_json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("tool 'web_search' conflicts with a client-supplied tool"));
        assert_eq!(provider_hits.load(Ordering::Relaxed), 0);
    }
}
