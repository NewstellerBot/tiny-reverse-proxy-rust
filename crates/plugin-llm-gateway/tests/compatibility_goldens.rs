#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};
    use serde::Deserialize;

    use plugin_llm_gateway::virtual_keys::VirtualKeys;
    use proxy_core::config::{
        AudioSurface, AudioSurfaceProtocol, BatchSurface, EmbeddingSurface,
        EmbeddingSurfaceProtocol, ImageSurface, ImageSurfaceProtocol, ProviderCommonConfig,
        ProviderFamily, ProviderFamilyConfig, ProviderKeyConfig, ProviderSurfaceCatalog,
        ResponsesSurface, ToolSurface,
    };
    use proxy_core::plugin::PluginChain;
    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream_async,
        TestProxyConfig,
    };

    #[derive(Debug, Deserialize)]
    struct GoldenFixture {
        scenario_id: String,
        request: GoldenRequest,
        provider_context: GoldenProviderContext,
        upstream_response: GoldenHttpResponse,
        expected_upstream: GoldenExpectation,
        expected_client: GoldenClientExpectation,
    }

    #[derive(Debug, Deserialize)]
    struct GoldenRequest {
        method: String,
        path: String,
        headers: BTreeMap<String, String>,
        body: serde_json::Value,
    }

    #[derive(Debug, Deserialize)]
    struct GoldenProviderContext {
        family: String,
        surfaced_protocol: String,
        request_mode: String,
    }

    #[derive(Debug, Deserialize)]
    struct GoldenHttpResponse {
        status: u16,
        headers: BTreeMap<String, String>,
        body: serde_json::Value,
    }

    #[derive(Debug, Deserialize)]
    struct GoldenExpectation {
        path: Option<String>,
        body_contains: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    struct GoldenClientExpectation {
        status: u16,
        body_contains: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct CapturedRequest {
        path: String,
        body_text: String,
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .unwrap()
            .to_path_buf()
    }

    fn fixture_path(relative: &str) -> PathBuf {
        repo_root()
            .join("tests")
            .join("reliability")
            .join("goldens")
            .join(relative)
    }

    fn load_fixture(relative: &str) -> GoldenFixture {
        serde_json::from_str(&fs::read_to_string(fixture_path(relative)).unwrap()).unwrap()
    }

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

    fn body_text(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        }
    }

    fn assert_contains_all(haystack: &str, expected: &[String]) {
        for needle in expected {
            assert!(
                haystack.contains(needle),
                "expected body to contain {needle:?}, got {haystack}"
            );
        }
    }

    async fn capture_upstream(
        fixture: &GoldenFixture,
        capture: Arc<Mutex<Vec<CapturedRequest>>>,
    ) -> String {
        let status = fixture.upstream_response.status;
        let headers = fixture.upstream_response.headers.clone();
        let body = body_text(&fixture.upstream_response.body);
        start_upstream_async(move |req: Request<Incoming>| {
            let capture = Arc::clone(&capture);
            let headers = headers.clone();
            let body = body.clone();
            async move {
                let path = req.uri().path().to_string();
                let body_bytes = req.into_body().collect().await.unwrap().to_bytes();
                capture.lock().unwrap().push(CapturedRequest {
                    path,
                    body_text: String::from_utf8_lossy(&body_bytes).into_owned(),
                });

                let mut builder = Response::builder().status(status);
                for (name, value) in &headers {
                    builder = builder.header(name, value);
                }
                builder.body(Full::new(Bytes::from(body))).unwrap()
            }
        })
        .await
    }

    fn build_request(fixture: &GoldenFixture, api_key: &str) -> Request<Full<Bytes>> {
        let mut builder = Request::builder()
            .method(fixture.request.method.as_str())
            .uri(fixture.request.path.as_str());
        for (name, value) in &fixture.request.headers {
            builder = builder.header(name, value);
        }
        builder
            .header("authorization", format!("Bearer {api_key}"))
            .body(Full::new(Bytes::from(body_text(&fixture.request.body))))
            .unwrap()
    }

    async fn configure_key(
        providers: &[ProviderKeyConfig],
        key_name: &str,
        primary_provider: &str,
    ) -> (VirtualKeys, String) {
        let vk = VirtualKeys::new(providers, &[], None);
        let (plaintext_key, _) = vk
            .create_key_for_project(
                Some("project-a"),
                key_name,
                primary_provider,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        (vk, plaintext_key)
    }

    #[tokio::test]
    async fn fixtures_parse_and_cover_expected_scenarios() {
        let fixtures = [
            "responses/non_streaming_structured_output.json",
            "responses/streaming_tool_event.json",
            "batches/native_supported_surface.json",
            "batches/reject_translated_only_target.json",
            "images/native_fallback_incompatible_field.json",
            "audio/native_fallback_incompatible_field.json",
            "embeddings/native_fallback_incompatible_input.json",
        ]
        .into_iter()
        .map(load_fixture)
        .collect::<Vec<_>>();

        assert_eq!(fixtures.len(), 7);
        for fixture in fixtures {
            assert!(!fixture.scenario_id.is_empty());
            assert!(!fixture.provider_context.family.is_empty());
            assert!(!fixture.provider_context.surfaced_protocol.is_empty());
            assert!(!fixture.provider_context.request_mode.is_empty());
            assert!(!fixture.request.method.is_empty());
            assert!(!fixture.request.path.is_empty());
        }
    }

    #[tokio::test]
    async fn responses_non_streaming_structured_output_golden() {
        let fixture = load_fixture("responses/non_streaming_structured_output.json");
        let capture = Arc::new(Mutex::new(Vec::new()));
        let upstream_addr = capture_upstream(&fixture, Arc::clone(&capture)).await;

        let providers = vec![canonical_provider(
            "responses-native",
            "sk-responses",
            format!("http://{upstream_addr}"),
            vec!["gpt-4.1-mini".to_string()],
            ProviderFamily::OpenAi,
            ProviderSurfaceCatalog {
                tools: Some(ToolSurface::OpenAi),
                responses: Some(ResponsesSurface::OpenAiCompatible),
                structured_output_json_schema: true,
                ..Default::default()
            },
        )];
        let (vk, plaintext_key) =
            configure_key(&providers, "responses-non-streaming", "responses-native").await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![upstream_addr]),
            TestProxyConfig {
                plugins: Some(Arc::new(PluginChain::new(vec![Box::new(vk)]))),
                ..Default::default()
            },
        )
        .await;

        let response = send_request(&proxy_addr, build_request(&fixture, &plaintext_key)).await;
        let status = response.status().as_u16();
        let body = response.collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&body);

        assert_eq!(status, fixture.expected_client.status);
        assert_contains_all(&body_text, &fixture.expected_client.body_contains);
        let captured = capture.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(
            Some(captured[0].path.as_str()),
            fixture.expected_upstream.path.as_deref()
        );
        assert_contains_all(
            &captured[0].body_text,
            &fixture.expected_upstream.body_contains,
        );
    }

    #[tokio::test]
    async fn responses_streaming_tool_event_golden() {
        let fixture = load_fixture("responses/streaming_tool_event.json");
        let capture = Arc::new(Mutex::new(Vec::new()));
        let upstream_addr = capture_upstream(&fixture, Arc::clone(&capture)).await;

        let providers = vec![canonical_provider(
            "responses-stream",
            "sk-responses-stream",
            format!("http://{upstream_addr}"),
            vec!["gpt-4.1-mini".to_string()],
            ProviderFamily::OpenAi,
            ProviderSurfaceCatalog {
                tools: Some(ToolSurface::OpenAi),
                responses: Some(ResponsesSurface::OpenAiCompatible),
                ..Default::default()
            },
        )];
        let (vk, plaintext_key) =
            configure_key(&providers, "responses-streaming", "responses-stream").await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![upstream_addr]),
            TestProxyConfig {
                plugins: Some(Arc::new(PluginChain::new(vec![Box::new(vk)]))),
                ..Default::default()
            },
        )
        .await;

        let response = send_request(&proxy_addr, build_request(&fixture, &plaintext_key)).await;
        let status = response.status().as_u16();
        let body = response.collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&body);

        assert_eq!(status, fixture.expected_client.status);
        assert_contains_all(&body_text, &fixture.expected_client.body_contains);
        let captured = capture.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_contains_all(
            &captured[0].body_text,
            &fixture.expected_upstream.body_contains,
        );
    }

    #[tokio::test]
    async fn batch_native_supported_surface_golden() {
        let fixture = load_fixture("batches/native_supported_surface.json");
        let standard_hits = Arc::new(AtomicUsize::new(0));
        let standard_addr = start_upstream_async({
            let standard_hits = Arc::clone(&standard_hits);
            move |_req| {
                let standard_hits = Arc::clone(&standard_hits);
                async move {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-standard")))
                        .unwrap()
                }
            }
        })
        .await;
        let capture = Arc::new(Mutex::new(Vec::new()));
        let batch_addr = capture_upstream(&fixture, Arc::clone(&capture)).await;

        let providers = vec![
            canonical_provider(
                "standard",
                "sk-standard",
                format!("http://{standard_addr}"),
                vec!["gpt-4o".to_string()],
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    ..Default::default()
                },
            ),
            canonical_provider(
                "batch-native",
                "sk-batch",
                format!("http://{batch_addr}"),
                vec!["gpt-4o".to_string()],
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    responses: Some(ResponsesSurface::OpenAiCompatible),
                    batches: Some(BatchSurface::OpenAiCompatible),
                    ..Default::default()
                },
            ),
        ];

        let (vk, plaintext_key) = configure_key(&providers, "batch-native", "standard").await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![standard_addr]),
            TestProxyConfig {
                plugins: Some(Arc::new(PluginChain::new(vec![Box::new(vk)]))),
                ..Default::default()
            },
        )
        .await;

        let response = send_request(&proxy_addr, build_request(&fixture, &plaintext_key)).await;
        let status = response.status().as_u16();
        let body = response.collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&body);

        assert_eq!(status, fixture.expected_client.status);
        assert_contains_all(&body_text, &fixture.expected_client.body_contains);
        assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
        let captured = capture.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(
            Some(captured[0].path.as_str()),
            fixture.expected_upstream.path.as_deref()
        );
        assert_contains_all(
            &captured[0].body_text,
            &fixture.expected_upstream.body_contains,
        );
    }

    #[tokio::test]
    async fn batch_reject_translated_only_target_golden() {
        let fixture = load_fixture("batches/reject_translated_only_target.json");
        let upstream_hits = Arc::new(AtomicUsize::new(0));
        let upstream_addr = start_upstream_async({
            let upstream_hits = Arc::clone(&upstream_hits);
            move |_req| {
                let upstream_hits = Arc::clone(&upstream_hits);
                async move {
                    upstream_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![canonical_provider(
            "responses-only",
            "sk-responses-only",
            format!("http://{upstream_addr}"),
            vec!["gpt-4o".to_string()],
            ProviderFamily::OpenAi,
            ProviderSurfaceCatalog {
                tools: Some(ToolSurface::OpenAi),
                responses: Some(ResponsesSurface::OpenAiCompatible),
                ..Default::default()
            },
        )];
        let (vk, plaintext_key) = configure_key(&providers, "batch-reject", "responses-only").await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![upstream_addr]),
            TestProxyConfig {
                plugins: Some(Arc::new(PluginChain::new(vec![Box::new(vk)]))),
                ..Default::default()
            },
        )
        .await;

        let response = send_request(&proxy_addr, build_request(&fixture, &plaintext_key)).await;
        let status = response.status().as_u16();
        let body = response.collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&body);

        assert_eq!(status, fixture.expected_client.status);
        assert_contains_all(&body_text, &fixture.expected_client.body_contains);
        assert_eq!(upstream_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn image_native_fallback_golden() {
        let fixture = load_fixture("images/native_fallback_incompatible_field.json");
        let translated_hits = Arc::new(AtomicUsize::new(0));
        let native_capture = Arc::new(Mutex::new(Vec::new()));
        let native_addr = capture_upstream(&fixture, Arc::clone(&native_capture)).await;
        let translated_addr = {
            let translated_hits = Arc::clone(&translated_hits);
            start_upstream_async(move |_req: Request<Incoming>| {
                let translated_hits = Arc::clone(&translated_hits);
                async move {
                    translated_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-translated")))
                        .unwrap()
                }
            })
            .await
        };

        let providers = vec![
            canonical_provider(
                "openrouter-images",
                "sk-openrouter",
                format!("http://{translated_addr}"),
                vec!["gpt-image-1".to_string()],
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
                    ..Default::default()
                },
            ),
            canonical_provider(
                "native-images",
                "sk-native-images",
                format!("http://{native_addr}"),
                vec!["gpt-image-1".to_string()],
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
                    ..Default::default()
                },
            ),
        ];
        let (vk, plaintext_key) =
            configure_key(&providers, "image-fallback", "openrouter-images").await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![translated_addr]),
            TestProxyConfig {
                plugins: Some(Arc::new(PluginChain::new(vec![Box::new(vk)]))),
                ..Default::default()
            },
        )
        .await;

        let response = send_request(&proxy_addr, build_request(&fixture, &plaintext_key)).await;
        let status = response.status().as_u16();
        let body = response.collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&body);

        assert_eq!(status, fixture.expected_client.status);
        assert_contains_all(&body_text, &fixture.expected_client.body_contains);
        assert_eq!(translated_hits.load(Ordering::Relaxed), 0);
        let captured = native_capture.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_contains_all(
            &captured[0].body_text,
            &fixture.expected_upstream.body_contains,
        );
    }

    #[tokio::test]
    async fn audio_native_fallback_golden() {
        let fixture = load_fixture("audio/native_fallback_incompatible_field.json");
        let translated_hits = Arc::new(AtomicUsize::new(0));
        let translated_addr = start_upstream_async({
            let translated_hits = Arc::clone(&translated_hits);
            move |_req: Request<Incoming>| {
                let translated_hits = Arc::clone(&translated_hits);
                async move {
                    translated_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(
                            b"unexpected-translated-audio",
                        )))
                        .unwrap()
                }
            }
        })
        .await;
        let native_capture = Arc::new(Mutex::new(Vec::new()));
        let native_addr = capture_upstream(&fixture, Arc::clone(&native_capture)).await;

        let providers = vec![
            canonical_provider(
                "openrouter-audio",
                "sk-openrouter-audio",
                format!("http://{translated_addr}"),
                vec!["gpt-4o-mini-tts".to_string()],
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
                    ..Default::default()
                },
            ),
            canonical_provider(
                "native-audio",
                "sk-native-audio",
                format!("http://{native_addr}"),
                vec!["gpt-4o-mini-tts".to_string()],
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
                    ..Default::default()
                },
            ),
        ];
        let (vk, plaintext_key) =
            configure_key(&providers, "audio-fallback", "openrouter-audio").await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![translated_addr]),
            TestProxyConfig {
                plugins: Some(Arc::new(PluginChain::new(vec![Box::new(vk)]))),
                ..Default::default()
            },
        )
        .await;

        let response = send_request(&proxy_addr, build_request(&fixture, &plaintext_key)).await;
        let status = response.status().as_u16();
        let body = response.collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&body);

        assert_eq!(status, fixture.expected_client.status);
        assert_contains_all(&body_text, &fixture.expected_client.body_contains);
        assert_eq!(translated_hits.load(Ordering::Relaxed), 0);
        let captured = native_capture.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_contains_all(
            &captured[0].body_text,
            &fixture.expected_upstream.body_contains,
        );
    }

    #[tokio::test]
    async fn embeddings_native_fallback_golden() {
        let fixture = load_fixture("embeddings/native_fallback_incompatible_input.json");
        let translated_hits = Arc::new(AtomicUsize::new(0));
        let translated_addr = start_upstream_async({
            let translated_hits = Arc::clone(&translated_hits);
            move |_req: Request<Incoming>| {
                let translated_hits = Arc::clone(&translated_hits);
                async move {
                    translated_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(
                            b"unexpected-translated-embedding",
                        )))
                        .unwrap()
                }
            }
        })
        .await;
        let native_capture = Arc::new(Mutex::new(Vec::new()));
        let native_addr = capture_upstream(&fixture, Arc::clone(&native_capture)).await;

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
                format!("http://{native_addr}"),
                vec!["text-embedding-3-large".to_string()],
                ProviderFamily::OpenAi,
                openai_embedding_surfaces(),
            ),
        ];
        let (vk, plaintext_key) =
            configure_key(&providers, "embedding-fallback", "gemini-embeddings").await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![translated_addr]),
            TestProxyConfig {
                plugins: Some(Arc::new(PluginChain::new(vec![Box::new(vk)]))),
                ..Default::default()
            },
        )
        .await;

        let response = send_request(&proxy_addr, build_request(&fixture, &plaintext_key)).await;
        let status = response.status().as_u16();
        let body = response.collect().await.unwrap().to_bytes();
        let body_text = String::from_utf8_lossy(&body);

        assert_eq!(status, fixture.expected_client.status);
        assert_contains_all(&body_text, &fixture.expected_client.body_contains);
        assert_eq!(translated_hits.load(Ordering::Relaxed), 0);
        let captured = native_capture.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_contains_all(
            &captured[0].body_text,
            &fixture.expected_upstream.body_contains,
        );
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
}
