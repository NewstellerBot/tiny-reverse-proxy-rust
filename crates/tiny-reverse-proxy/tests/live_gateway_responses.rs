#![cfg(feature = "plugin-llm-gateway")]

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::BodyExt;
    use http_body_util::Full;
    use hyper::Request;
    use hyper::StatusCode;
    use proxy_core::config::{
        ProviderFamilyConfig, ProviderKeyConfig, ProviderSurfaceCatalog, ResponsesSurface,
    };
    use proxy_core::plugin::PluginChain;
    use serde_json::{json, Value};
    use tempfile::NamedTempFile;

    use trp_test_support::{
        catch_all_router, openai_api_key, openai_organization, openai_project,
        openai_responses_model, openai_responses_timeout, send_request, start_proxy_with_config,
        TestProxyConfig,
    };

    fn provider_config() -> ProviderKeyConfig {
        let family = ProviderFamilyConfig::OpenAi {
            surfaces: ProviderSurfaceCatalog {
                responses: Some(ResponsesSurface::OpenAiCompatible),
                ..ProviderSurfaceCatalog::default()
            },
        };
        let surfaces = family.surfaces().clone();

        ProviderKeyConfig {
            name: "openai-live".to_string(),
            api_key: openai_api_key("run the live gateway smoke tests"),
            base_url: "https://api.openai.com".to_string(),
            models: vec![openai_responses_model()],
            api_key_header: "authorization".to_string(),
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

    async fn setup_live_gateway() -> (String, String, NamedTempFile) {
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let providers = vec![provider_config()];

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &[],
            Some(&store_url),
            &providers,
            &[],
            plugin_llm_gateway::CreatePluginsOptions::default(),
            None,
        )
        .await
        .expect("create live gateway plugins");

        let (plaintext_key, _) = api
            .create_virtual_key_with_runtime_policy(
                Some("live-smoke"),
                "live-openai-responses",
                "openai-live",
                None,
                None,
                None,
                None,
                Some(vec![openai_responses_model()]),
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create live virtual key");

        let plugins = Arc::new(PluginChain::new(plugins));
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec!["https://api.openai.com".to_string()]),
            TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            },
        )
        .await;

        (proxy_addr, plaintext_key, temp_db)
    }

    async fn post_gateway_responses(
        proxy_addr: &str,
        virtual_key: &str,
        body_json: Value,
    ) -> (StatusCode, String) {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", format!("Bearer {virtual_key}"))
            .header("content-type", "application/json");

        if let Some(org) = openai_organization() {
            builder = builder.header("OpenAI-Organization", org);
        }
        if let Some(project) = openai_project() {
            builder = builder.header("OpenAI-Project", project);
        }

        let req = builder
            .body(Full::new(Bytes::from(body_json.to_string())))
            .unwrap();
        let resp = send_request(proxy_addr, req).await;
        let status = resp.status();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    fn response_items_text(items: &[Value]) -> Option<String> {
        let mut text = String::new();
        for item in items {
            for part in item
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(fragment) = part.get("text").and_then(Value::as_str) {
                    text.push_str(fragment);
                }
            }
        }

        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    fn responses_output_text(value: &Value) -> Option<String> {
        if let Some(text) = value.get("output_text").and_then(Value::as_str) {
            if !text.trim().is_empty() {
                return Some(text.to_string());
            }
        }

        if let Some(items) = value.get("output").and_then(Value::as_array) {
            if let Some(text) = response_items_text(items) {
                return Some(text);
            }
        }

        value.get("response").and_then(responses_output_text)
    }

    fn parse_sse_events(body: &str) -> Vec<Value> {
        body.split("\n\n")
            .filter_map(|chunk| {
                let mut data_lines = Vec::new();
                for line in chunk.lines() {
                    if let Some(data) = line.trim_end().strip_prefix("data:") {
                        data_lines.push(data.trim().to_string());
                    }
                }

                if data_lines.is_empty() {
                    return None;
                }

                let payload = data_lines.join("\n");
                if payload == "[DONE]" {
                    return None;
                }

                Some(
                    serde_json::from_str::<Value>(&payload).unwrap_or_else(|e| {
                        panic!("invalid SSE JSON payload: {e}; payload={payload}")
                    }),
                )
            })
            .collect()
    }

    fn responses_sse_output_text(events: &[Value]) -> Option<String> {
        let mut text = String::new();

        for event in events {
            match event.get("type").and_then(Value::as_str).unwrap_or("") {
                "response.output_text.delta" | "response.text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                        text.push_str(delta);
                    }
                }
                "response.output_text.done" | "response.text.done" => {
                    if text.trim().is_empty() {
                        if let Some(done_text) = event.get("text").and_then(Value::as_str) {
                            text.push_str(done_text);
                        }
                    }
                }
                "response.completed" | "response.done" => {
                    if text.trim().is_empty() {
                        if let Some(done_text) = responses_output_text(event) {
                            text.push_str(&done_text);
                        }
                    }
                }
                "error" => panic!("Responses API returned error event: {}", event),
                _ => {}
            }
        }

        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and live OpenAI Responses API access"]
    async fn gateway_virtual_key_responses_proxy_can_generate_text_response() {
        let timeout = openai_responses_timeout();
        let (proxy_addr, virtual_key, _temp_db) =
            tokio::time::timeout(timeout, setup_live_gateway())
                .await
                .expect("timed out starting live gateway");

        let (status, body) = tokio::time::timeout(
            timeout,
            post_gateway_responses(
                &proxy_addr,
                &virtual_key,
                json!({
                    "model": openai_responses_model(),
                    "input": "Reply with exactly PONG and nothing else."
                }),
            ),
        )
        .await
        .expect("timed out running live gateway non-streaming request");

        assert_eq!(
            status,
            StatusCode::OK,
            "unexpected status with body: {body}"
        );
        let response_json: Value = serde_json::from_str(&body)
            .unwrap_or_else(|e| panic!("invalid Responses JSON body: {e}; body={body}"));
        let output_text = responses_output_text(&response_json)
            .unwrap_or_else(|| panic!("Responses body missing output text: {response_json}"));
        eprintln!("Gateway Responses text response: {output_text}");
        assert!(
            output_text.to_ascii_uppercase().contains("PONG"),
            "expected response to contain PONG, got: {output_text}"
        );
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and live OpenAI Responses API access"]
    async fn gateway_virtual_key_responses_proxy_can_stream_text_response() {
        let timeout = openai_responses_timeout();
        let (proxy_addr, virtual_key, _temp_db) =
            tokio::time::timeout(timeout, setup_live_gateway())
                .await
                .expect("timed out starting live gateway");

        let (status, body) = tokio::time::timeout(
            timeout,
            post_gateway_responses(
                &proxy_addr,
                &virtual_key,
                json!({
                    "model": openai_responses_model(),
                    "input": "Reply with exactly PONG and nothing else.",
                    "stream": true
                }),
            ),
        )
        .await
        .expect("timed out running live gateway streaming request");

        assert_eq!(
            status,
            StatusCode::OK,
            "unexpected status with body: {body}"
        );
        assert!(
            body.contains("response.output_text.delta")
                || body.contains("response.completed")
                || body.contains("response.done"),
            "expected streaming SSE events, got: {body}"
        );

        let events = parse_sse_events(&body);
        assert!(
            !events.is_empty(),
            "expected at least one SSE event in body: {body}"
        );
        let output_text = responses_sse_output_text(&events)
            .unwrap_or_else(|| panic!("Responses stream missing output text: {body}"));
        eprintln!("Gateway Responses streaming text response: {output_text}");
        assert!(
            output_text.to_ascii_uppercase().contains("PONG"),
            "expected stream to contain PONG, got: {output_text}"
        );
    }
}
