#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use plugin_llm_gateway::api::LlmGatewayApi;
    use plugin_llm_gateway::cost_tracker;
    use plugin_llm_gateway::governance::current_timestamp_string;
    use plugin_llm_gateway::provider_failover;
    use plugin_llm_gateway::rate_limiter;
    use plugin_llm_gateway::store::{
        self, GatewayStore, GovernanceChangeRecord, KeyModelUsageRecord, KeyUsageRecord,
        ManagedProviderRecord, ModelCostRecord, ProjectDatasetItemRecord, ProjectDatasetRecord,
        ProjectEvalRunItemRecord, ProjectEvalRunRecord, ProjectPromptRecord,
        ProjectPromptRolloutRecord, ProjectRolloutPolicyRecord, RequestLogEntry,
        SessionEventRecord, SessionListQuery, SessionRecord,
    };

    // -----------------------------------------------------------------------
    // Store CRUD tests (in-memory SQLite)
    // -----------------------------------------------------------------------

    mod store_tests {
        use super::*;

        async fn new_store() -> store::Store {
            store::connect("sqlite::memory:").await.unwrap()
        }

        #[tokio::test]
        async fn usage_crud() {
            let store = new_store().await;

            // Initially empty.
            assert!(store.get_all_usage().await.unwrap().is_empty());
            assert!(store.get_usage("key1").await.unwrap().is_none());

            // Insert.
            let usage = KeyUsageRecord {
                total_input_tokens: 1000,
                total_output_tokens: 500,
                total_cost: 0.05,
            };
            store.upsert_usage("key1", &usage).await.unwrap();

            // Read back.
            let got = store.get_usage("key1").await.unwrap().unwrap();
            assert_eq!(got.total_input_tokens, 1000);
            assert_eq!(got.total_output_tokens, 500);
            assert!((got.total_cost - 0.05).abs() < 1e-9);

            // Update (upsert).
            let updated = KeyUsageRecord {
                total_input_tokens: 2000,
                total_output_tokens: 1000,
                total_cost: 0.10,
            };
            store.upsert_usage("key1", &updated).await.unwrap();
            let got = store.get_usage("key1").await.unwrap().unwrap();
            assert_eq!(got.total_input_tokens, 2000);

            // List all.
            store
                .upsert_usage(
                    "key2",
                    &KeyUsageRecord {
                        total_input_tokens: 100,
                        total_output_tokens: 50,
                        total_cost: 0.01,
                    },
                )
                .await
                .unwrap();
            assert_eq!(store.get_all_usage().await.unwrap().len(), 2);

            // Delete one.
            assert!(store.delete_usage("key1").await.unwrap());
            assert!(!store.delete_usage("nonexistent").await.unwrap());
            assert_eq!(store.get_all_usage().await.unwrap().len(), 1);

            // Delete all.
            store.delete_all_usage().await.unwrap();
            assert!(store.get_all_usage().await.unwrap().is_empty());
        }

        #[tokio::test]
        async fn model_cost_crud() {
            let store = new_store().await;

            // Initially empty.
            assert!(store.get_all_model_costs().await.unwrap().is_empty());
            assert!(store.get_model_cost("gpt-4").await.unwrap().is_none());

            // Insert.
            let cost = ModelCostRecord {
                input_cost_per_1k: 0.03,
                output_cost_per_1k: 0.06,
            };
            store.upsert_model_cost("gpt-4", &cost).await.unwrap();

            // Read back.
            let got = store.get_model_cost("gpt-4").await.unwrap().unwrap();
            assert!((got.input_cost_per_1k - 0.03).abs() < 1e-9);
            assert!((got.output_cost_per_1k - 0.06).abs() < 1e-9);

            // Upsert update.
            let updated = ModelCostRecord {
                input_cost_per_1k: 0.05,
                output_cost_per_1k: 0.10,
            };
            store.upsert_model_cost("gpt-4", &updated).await.unwrap();
            let got = store.get_model_cost("gpt-4").await.unwrap().unwrap();
            assert!((got.input_cost_per_1k - 0.05).abs() < 1e-9);

            // List all.
            store
                .upsert_model_cost(
                    "claude-3",
                    &ModelCostRecord {
                        input_cost_per_1k: 0.015,
                        output_cost_per_1k: 0.075,
                    },
                )
                .await
                .unwrap();
            assert_eq!(store.get_all_model_costs().await.unwrap().len(), 2);

            // Delete one.
            assert!(store.delete_model_cost("gpt-4").await.unwrap());
            assert!(!store.delete_model_cost("nonexistent").await.unwrap());
            assert_eq!(store.get_all_model_costs().await.unwrap().len(), 1);
        }

        #[tokio::test]
        async fn per_model_usage_crud() {
            let store = new_store().await;

            // Initially empty.
            assert!(store.get_all_per_model_usage().await.unwrap().is_empty());

            // Insert.
            let record = KeyModelUsageRecord {
                api_key: "key1".to_string(),
                model: "gpt-4o".to_string(),
                total_input_tokens: 1000,
                total_output_tokens: 500,
                total_cost: 0.05,
            };
            store.upsert_per_model_usage(&record).await.unwrap();

            let all = store.get_all_per_model_usage().await.unwrap();
            assert_eq!(all.len(), 1);
            assert_eq!(all[0].api_key, "key1");
            assert_eq!(all[0].model, "gpt-4o");
            assert_eq!(all[0].total_input_tokens, 1000);

            // Upsert same key+model updates values.
            let updated = KeyModelUsageRecord {
                api_key: "key1".to_string(),
                model: "gpt-4o".to_string(),
                total_input_tokens: 2000,
                total_output_tokens: 1000,
                total_cost: 0.10,
            };
            store.upsert_per_model_usage(&updated).await.unwrap();
            let all = store.get_all_per_model_usage().await.unwrap();
            assert_eq!(all.len(), 1);
            assert_eq!(all[0].total_input_tokens, 2000);

            // Different model for same key = new row.
            let record2 = KeyModelUsageRecord {
                api_key: "key1".to_string(),
                model: "claude-sonnet".to_string(),
                total_input_tokens: 500,
                total_output_tokens: 250,
                total_cost: 0.03,
            };
            store.upsert_per_model_usage(&record2).await.unwrap();
            assert_eq!(store.get_all_per_model_usage().await.unwrap().len(), 2);

            // Delete all.
            store.delete_all_per_model_usage().await.unwrap();
            assert!(store.get_all_per_model_usage().await.unwrap().is_empty());
        }

        #[tokio::test]
        async fn request_log_crud() {
            let store = new_store().await;

            // Initially empty.
            let logs = store.get_request_logs(None, None, None, 100).await.unwrap();
            assert!(logs.is_empty());

            // Append entries.
            let entries = vec![
                RequestLogEntry {
                    timestamp_unix: 1000,
                    api_key: "key1".to_string(),
                    project_id: Some("legacy".to_string()),
                    session_id: Some("session-a".to_string()),
                    metadata_json: None,
                    custom_cost_json: None,
                    custom_cost_applied: false,
                    provider_name: Some("openai".to_string()),
                    prompt_name: Some("support".to_string()),
                    prompt_version: Some("v1".to_string()),
                    prompt_environment: Some("prod".to_string()),
                    model: Some("gpt-4o".to_string()),
                    input_tokens: 100,
                    output_tokens: 50,
                    cost: 0.01,
                    is_streaming: false,
                    safety_mode: None,
                    safety_matches: None,
                    semantic_policy_version: None,
                    semantic_index_state: None,
                    semantic_degraded_reason: None,
                    semantic_findings: None,
                    tool_trace: None,
                },
                RequestLogEntry {
                    timestamp_unix: 2000,
                    api_key: "key1".to_string(),
                    project_id: Some("legacy".to_string()),
                    session_id: Some("session-a".to_string()),
                    metadata_json: None,
                    custom_cost_json: None,
                    custom_cost_applied: false,
                    provider_name: Some("anthropic".to_string()),
                    prompt_name: None,
                    prompt_version: None,
                    prompt_environment: None,
                    model: Some("claude-sonnet".to_string()),
                    input_tokens: 200,
                    output_tokens: 100,
                    cost: 0.02,
                    is_streaming: true,
                    safety_mode: None,
                    safety_matches: None,
                    semantic_policy_version: Some("sem-v2".to_string()),
                    semantic_index_state: Some("ready".to_string()),
                    semantic_degraded_reason: None,
                    semantic_findings: Some(
                        r#"[{"chunk_path":"$.messages[0].content","topic_id":"layoffs"}]"#
                            .to_string(),
                    ),
                    tool_trace: Some(
                        r#"{"calls":[{"tool_name":"web_search","executor_kind":"web_search","status":"success","error_code":null}]}"#
                            .to_string(),
                    ),
                },
                RequestLogEntry {
                    timestamp_unix: 3000,
                    api_key: "key2".to_string(),
                    project_id: Some("legacy".to_string()),
                    session_id: Some("session-b".to_string()),
                    metadata_json: None,
                    custom_cost_json: None,
                    custom_cost_applied: false,
                    provider_name: Some("openai".to_string()),
                    prompt_name: Some("analysis".to_string()),
                    prompt_version: Some("v3".to_string()),
                    prompt_environment: Some("staging".to_string()),
                    model: Some("gpt-4o".to_string()),
                    input_tokens: 300,
                    output_tokens: 150,
                    cost: 0.03,
                    is_streaming: false,
                    safety_mode: None,
                    safety_matches: None,
                    semantic_policy_version: None,
                    semantic_index_state: None,
                    semantic_degraded_reason: Some("timeout".to_string()),
                    semantic_findings: Some("[]".to_string()),
                    tool_trace: None,
                },
            ];
            store.append_request_logs(&entries).await.unwrap();

            // Get all.
            let logs = store.get_request_logs(None, None, None, 100).await.unwrap();
            assert_eq!(logs.len(), 3);
            // Ordered by timestamp DESC.
            assert_eq!(logs[0].timestamp_unix, 3000);
            assert_eq!(logs[2].timestamp_unix, 1000);

            // Filter by api_key.
            let logs = store
                .get_request_logs(Some("key1"), None, None, 100)
                .await
                .unwrap();
            assert_eq!(logs.len(), 2);

            // Filter by model.
            let logs = store
                .get_request_logs(None, Some("gpt-4o"), None, 100)
                .await
                .unwrap();
            assert_eq!(logs.len(), 2);

            // Filter by both.
            let logs = store
                .get_request_logs(Some("key1"), Some("gpt-4o"), None, 100)
                .await
                .unwrap();
            assert_eq!(logs.len(), 1);
            assert_eq!(logs[0].api_key, "key1");
            assert_eq!(logs[0].model.as_deref(), Some("gpt-4o"));
            assert_eq!(logs[0].prompt_name.as_deref(), Some("support"));
            assert_eq!(logs[0].prompt_version.as_deref(), Some("v1"));
            assert_eq!(logs[0].prompt_environment.as_deref(), Some("prod"));

            // Limit.
            let logs = store.get_request_logs(None, None, None, 2).await.unwrap();
            assert_eq!(logs.len(), 2);

            // Verify streaming flag round-trip.
            let logs = store
                .get_request_logs(Some("key1"), Some("claude-sonnet"), None, 100)
                .await
                .unwrap();
            assert_eq!(logs.len(), 1);
            assert!(logs[0].is_streaming);
            assert_eq!(logs[0].semantic_policy_version.as_deref(), Some("sem-v2"));
            assert_eq!(logs[0].semantic_index_state.as_deref(), Some("ready"));
            assert_eq!(
                logs[0].tool_trace.as_deref(),
                Some(
                    r#"{"calls":[{"tool_name":"web_search","executor_kind":"web_search","status":"success","error_code":null}]}"#
                )
            );

            let logs = store
                .get_request_logs_for_session("session-a", Some("legacy"), 100)
                .await
                .unwrap();
            assert_eq!(logs.len(), 2);
            assert_eq!(logs[0].timestamp_unix, 2000);
            assert_eq!(logs[1].session_id.as_deref(), Some("session-a"));
        }

        #[tokio::test]
        async fn session_state_round_trip() {
            let store = new_store().await;
            let record = SessionRecord {
                session_id: "session-state-a".to_string(),
                project_id: Some("legacy".to_string()),
                project_ids_json: Some(r#"["legacy"]"#.to_string()),
                first_request_unix: Some(1000),
                last_request_unix: Some(2000),
                updated_at_unix: 2500,
                request_count: 2,
                streaming_request_count: 1,
                total_input_tokens: 300,
                total_output_tokens: 150,
                total_cost: 0.03,
                providers_json: Some(r#"["anthropic","openai"]"#.to_string()),
                models_json: Some(r#"["claude-sonnet","gpt-4o"]"#.to_string()),
                prompt_names_json: Some(r#"["analysis","support"]"#.to_string()),
                prompt_versions_json: Some(r#"["support@v1"]"#.to_string()),
                tool_names_json: Some(r#"["web_search"]"#.to_string()),
                latest_request_json: Some(
                    serde_json::json!({
                        "timestamp_unix": 2000,
                        "provider_name": "anthropic",
                        "model": "claude-sonnet"
                    })
                    .to_string(),
                ),
                safety_event_count: 1,
                semantic_event_count: 1,
                semantic_degraded_count: 0,
                tool_call_count: 2,
                tool_error_count: 1,
                status: Some("active".to_string()),
                owner_id: Some("worker-a".to_string()),
                owner_acquired_at_unix: Some(2300),
                last_transition_at_unix: Some(2400),
                last_transition_reason: Some("resumed".to_string()),
                last_heartbeat_unix: Some(2450),
                lease_expires_at_unix: Some(2600),
                cancel_requested_at_unix: Some(2550),
                cancel_requested_by: Some("operator-a".to_string()),
                cancel_reason: Some("stop after response".to_string()),
                handoff_target_owner_id: Some("worker-b".to_string()),
                handoff_requested_at_unix: Some(2575),
                handoff_reason: Some("handoff to worker-b".to_string()),
                state_json: Some(serde_json::json!({"turn": 2}).to_string()),
                metadata_json: Some(serde_json::json!({"owner": "qa"}).to_string()),
            };

            store.upsert_session(&record).await.unwrap();
            let fetched = store
                .get_session("session-state-a")
                .await
                .unwrap()
                .expect("session record");
            assert_eq!(fetched.project_id.as_deref(), Some("legacy"));
            assert_eq!(fetched.request_count, 2);
            assert_eq!(fetched.streaming_request_count, 1);
            assert_eq!(fetched.status.as_deref(), Some("active"));
            assert_eq!(fetched.owner_id.as_deref(), Some("worker-a"));
            assert_eq!(fetched.owner_acquired_at_unix, Some(2300));
            assert_eq!(fetched.last_transition_at_unix, Some(2400));
            assert_eq!(fetched.last_transition_reason.as_deref(), Some("resumed"));
            assert_eq!(fetched.last_heartbeat_unix, Some(2450));
            assert_eq!(fetched.lease_expires_at_unix, Some(2600));
            assert_eq!(fetched.cancel_requested_at_unix, Some(2550));
            assert_eq!(fetched.cancel_requested_by.as_deref(), Some("operator-a"));
            assert_eq!(
                fetched.cancel_reason.as_deref(),
                Some("stop after response")
            );
            assert_eq!(fetched.handoff_target_owner_id.as_deref(), Some("worker-b"));
            assert_eq!(fetched.handoff_requested_at_unix, Some(2575));
            assert_eq!(
                fetched.handoff_reason.as_deref(),
                Some("handoff to worker-b")
            );
            assert_eq!(fetched.state_json.as_deref(), Some(r#"{"turn":2}"#));
            assert_eq!(fetched.metadata_json.as_deref(), Some(r#"{"owner":"qa"}"#));
        }

        #[tokio::test]
        async fn list_sessions_filters_by_project_owner_status_and_time() {
            let store = new_store().await;

            let base = |session_id: &str,
                        project_id: Option<&str>,
                        project_ids_json: Option<&str>,
                        updated_at_unix: i64,
                        status: Option<&str>,
                        owner_id: Option<&str>| SessionRecord {
                session_id: session_id.to_string(),
                project_id: project_id.map(ToString::to_string),
                project_ids_json: project_ids_json.map(ToString::to_string),
                first_request_unix: Some(updated_at_unix - 10),
                last_request_unix: Some(updated_at_unix - 1),
                updated_at_unix,
                request_count: 1,
                streaming_request_count: 0,
                total_input_tokens: 10,
                total_output_tokens: 5,
                total_cost: 0.01,
                providers_json: None,
                models_json: None,
                prompt_names_json: None,
                prompt_versions_json: None,
                tool_names_json: None,
                latest_request_json: None,
                safety_event_count: 0,
                semantic_event_count: 0,
                semantic_degraded_count: 0,
                tool_call_count: 0,
                tool_error_count: 0,
                status: status.map(ToString::to_string),
                owner_id: owner_id.map(ToString::to_string),
                owner_acquired_at_unix: owner_id.map(|_| updated_at_unix - 5),
                last_transition_at_unix: Some(updated_at_unix - 5),
                last_transition_reason: Some("claimed".to_string()),
                last_heartbeat_unix: owner_id.map(|_| updated_at_unix - 2),
                lease_expires_at_unix: owner_id.map(|_| updated_at_unix + 60),
                cancel_requested_at_unix: None,
                cancel_requested_by: None,
                cancel_reason: None,
                handoff_target_owner_id: None,
                handoff_requested_at_unix: None,
                handoff_reason: None,
                state_json: None,
                metadata_json: None,
            };

            for record in [
                base(
                    "session-multi-project",
                    None,
                    Some(r#"["project-a","project-b"]"#),
                    70,
                    Some("active"),
                    Some("worker-a"),
                ),
                base(
                    "session-project-a-paused",
                    Some("project-a"),
                    Some(r#"["project-a"]"#),
                    80,
                    Some("paused"),
                    Some("worker-b"),
                ),
                base(
                    "session-project-a-active",
                    Some("project-a"),
                    Some(r#"["project-a"]"#),
                    90,
                    Some("active"),
                    Some("worker-a"),
                ),
                base(
                    "session-project-b-active",
                    Some("project-b"),
                    Some(r#"["project-b"]"#),
                    100,
                    Some("active"),
                    Some("worker-a"),
                ),
            ] {
                store.upsert_session(&record).await.unwrap();
            }

            let project_a_sessions = store
                .list_sessions(&SessionListQuery {
                    project_id: Some("project-a".to_string()),
                    limit: 10,
                    ..Default::default()
                })
                .await
                .unwrap();
            let project_a_ids = project_a_sessions
                .iter()
                .map(|record| record.session_id.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                project_a_ids,
                vec![
                    "session-project-a-active",
                    "session-project-a-paused",
                    "session-multi-project",
                ]
            );

            let filtered = store
                .list_sessions(&SessionListQuery {
                    project_id: Some("project-a".to_string()),
                    status: Some("active".to_string()),
                    owner_id: Some("worker-a".to_string()),
                    updated_after_unix: Some(75),
                    limit: 10,
                })
                .await
                .unwrap();
            let filtered_ids = filtered
                .iter()
                .map(|record| record.session_id.as_str())
                .collect::<Vec<_>>();
            assert_eq!(filtered_ids, vec!["session-project-a-active"]);
        }

        #[tokio::test]
        async fn session_events_round_trip_and_filter_by_after_seq() {
            let store = new_store().await;

            let claimed_seq = store
                .append_session_event(&SessionEventRecord {
                    event_seq: 0,
                    session_id: "session-events".to_string(),
                    project_id: Some("project-a".to_string()),
                    event_kind: "claimed".to_string(),
                    actor_id: Some("worker-a".to_string()),
                    reason: Some("claimed".to_string()),
                    payload_json: Some(serde_json::json!({"status":"active"}).to_string()),
                    created_at_unix: 10,
                })
                .await
                .unwrap();
            let cancel_seq = store
                .append_session_event(&SessionEventRecord {
                    event_seq: 0,
                    session_id: "session-events".to_string(),
                    project_id: Some("project-a".to_string()),
                    event_kind: "cancel_requested".to_string(),
                    actor_id: Some("operator-a".to_string()),
                    reason: Some("stop".to_string()),
                    payload_json: Some(serde_json::json!({"reason":"stop"}).to_string()),
                    created_at_unix: 20,
                })
                .await
                .unwrap();

            assert!(claimed_seq > 0);
            assert!(cancel_seq > claimed_seq);

            let events = store
                .get_session_events("session-events", None, 10)
                .await
                .unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(events[0].event_kind, "claimed");
            assert_eq!(events[1].event_kind, "cancel_requested");

            let later_events = store
                .get_session_events("session-events", Some(claimed_seq), 10)
                .await
                .unwrap();
            assert_eq!(later_events.len(), 1);
            assert_eq!(later_events[0].event_seq, cancel_seq);
            assert_eq!(later_events[0].actor_id.as_deref(), Some("operator-a"));
        }

        #[tokio::test]
        async fn list_sessions_for_recovery_returns_only_recoverable_sessions() {
            let store = new_store().await;

            let records = vec![
                SessionRecord {
                    session_id: "recover-me".to_string(),
                    project_id: Some("project-a".to_string()),
                    project_ids_json: Some(r#"["project-a"]"#.to_string()),
                    first_request_unix: Some(1),
                    last_request_unix: Some(2),
                    updated_at_unix: 2,
                    request_count: 1,
                    streaming_request_count: 0,
                    total_input_tokens: 0,
                    total_output_tokens: 0,
                    total_cost: 0.0,
                    providers_json: None,
                    models_json: None,
                    prompt_names_json: None,
                    prompt_versions_json: None,
                    tool_names_json: None,
                    latest_request_json: None,
                    safety_event_count: 0,
                    semantic_event_count: 0,
                    semantic_degraded_count: 0,
                    tool_call_count: 0,
                    tool_error_count: 0,
                    status: Some("active".to_string()),
                    owner_id: Some("worker-a".to_string()),
                    owner_acquired_at_unix: Some(1),
                    last_transition_at_unix: Some(1),
                    last_transition_reason: Some("claimed".to_string()),
                    last_heartbeat_unix: Some(2),
                    lease_expires_at_unix: Some(10),
                    cancel_requested_at_unix: None,
                    cancel_requested_by: None,
                    cancel_reason: None,
                    handoff_target_owner_id: None,
                    handoff_requested_at_unix: None,
                    handoff_reason: None,
                    state_json: None,
                    metadata_json: None,
                },
                SessionRecord {
                    session_id: "cancel-me".to_string(),
                    project_id: Some("project-a".to_string()),
                    project_ids_json: Some(r#"["project-a"]"#.to_string()),
                    first_request_unix: Some(1),
                    last_request_unix: Some(2),
                    updated_at_unix: 2,
                    request_count: 1,
                    streaming_request_count: 0,
                    total_input_tokens: 0,
                    total_output_tokens: 0,
                    total_cost: 0.0,
                    providers_json: None,
                    models_json: None,
                    prompt_names_json: None,
                    prompt_versions_json: None,
                    tool_names_json: None,
                    latest_request_json: None,
                    safety_event_count: 0,
                    semantic_event_count: 0,
                    semantic_degraded_count: 0,
                    tool_call_count: 0,
                    tool_error_count: 0,
                    status: Some("paused".to_string()),
                    owner_id: None,
                    owner_acquired_at_unix: None,
                    last_transition_at_unix: Some(1),
                    last_transition_reason: Some("waiting".to_string()),
                    last_heartbeat_unix: None,
                    lease_expires_at_unix: None,
                    cancel_requested_at_unix: Some(8),
                    cancel_requested_by: Some("operator-a".to_string()),
                    cancel_reason: Some("stop".to_string()),
                    handoff_target_owner_id: None,
                    handoff_requested_at_unix: None,
                    handoff_reason: None,
                    state_json: None,
                    metadata_json: None,
                },
                SessionRecord {
                    session_id: "ignore-active-owner".to_string(),
                    project_id: Some("project-a".to_string()),
                    project_ids_json: Some(r#"["project-a"]"#.to_string()),
                    first_request_unix: Some(1),
                    last_request_unix: Some(2),
                    updated_at_unix: 2,
                    request_count: 1,
                    streaming_request_count: 0,
                    total_input_tokens: 0,
                    total_output_tokens: 0,
                    total_cost: 0.0,
                    providers_json: None,
                    models_json: None,
                    prompt_names_json: None,
                    prompt_versions_json: None,
                    tool_names_json: None,
                    latest_request_json: None,
                    safety_event_count: 0,
                    semantic_event_count: 0,
                    semantic_degraded_count: 0,
                    tool_call_count: 0,
                    tool_error_count: 0,
                    status: Some("active".to_string()),
                    owner_id: Some("worker-b".to_string()),
                    owner_acquired_at_unix: Some(1),
                    last_transition_at_unix: Some(1),
                    last_transition_reason: Some("claimed".to_string()),
                    last_heartbeat_unix: Some(2),
                    lease_expires_at_unix: Some(30),
                    cancel_requested_at_unix: None,
                    cancel_requested_by: None,
                    cancel_reason: None,
                    handoff_target_owner_id: None,
                    handoff_requested_at_unix: None,
                    handoff_reason: None,
                    state_json: None,
                    metadata_json: None,
                },
                SessionRecord {
                    session_id: "ignore-terminal".to_string(),
                    project_id: Some("project-a".to_string()),
                    project_ids_json: Some(r#"["project-a"]"#.to_string()),
                    first_request_unix: Some(1),
                    last_request_unix: Some(2),
                    updated_at_unix: 2,
                    request_count: 1,
                    streaming_request_count: 0,
                    total_input_tokens: 0,
                    total_output_tokens: 0,
                    total_cost: 0.0,
                    providers_json: None,
                    models_json: None,
                    prompt_names_json: None,
                    prompt_versions_json: None,
                    tool_names_json: None,
                    latest_request_json: None,
                    safety_event_count: 0,
                    semantic_event_count: 0,
                    semantic_degraded_count: 0,
                    tool_call_count: 0,
                    tool_error_count: 0,
                    status: Some("completed".to_string()),
                    owner_id: Some("worker-c".to_string()),
                    owner_acquired_at_unix: Some(1),
                    last_transition_at_unix: Some(1),
                    last_transition_reason: Some("done".to_string()),
                    last_heartbeat_unix: Some(2),
                    lease_expires_at_unix: Some(3),
                    cancel_requested_at_unix: None,
                    cancel_requested_by: None,
                    cancel_reason: None,
                    handoff_target_owner_id: None,
                    handoff_requested_at_unix: None,
                    handoff_reason: None,
                    state_json: None,
                    metadata_json: None,
                },
            ];

            for record in records {
                store.upsert_session(&record).await.unwrap();
            }

            let mut sessions = store.list_sessions_for_recovery(10, 10).await.unwrap();
            sessions.sort();
            assert_eq!(
                sessions,
                vec!["cancel-me".to_string(), "recover-me".to_string()]
            );
        }

        #[tokio::test]
        async fn managed_provider_crud() {
            let store = new_store().await;
            let record = ManagedProviderRecord {
                name: "beta".to_string(),
                enabled: true,
                api_key_env: Some("TRP_BETA_KEY".to_string()),
                base_url: Some("https://beta.example.com".to_string()),
                models_json: Some(r#"["gpt-4o-mini"]"#.to_string()),
                api_key_header: Some("authorization".to_string()),
                timeout_secs: Some(42),
                family: Some("openai".to_string()),
                surfaces_json: Some(
                    r#"{"tools":"openai","responses":"openai_compatible","reasoning":true,"files":"openai_compatible"}"#
                        .to_string(),
                ),
                routing_metadata_json: Some(
                    r#"{"data_collection":"deny","zdr":true,"distillable_text":false,"quantizations":["fp8"],"supported_parameter_families":["reasoning"]}"#
                        .to_string(),
                ),
                created_at: "1".to_string(),
                updated_at: "2".to_string(),
            };

            store.upsert_managed_provider(&record).await.unwrap();

            let loaded = store
                .get_managed_provider("beta")
                .await
                .unwrap()
                .expect("managed provider");
            assert!(loaded.enabled);
            assert_eq!(loaded.api_key_env.as_deref(), Some("TRP_BETA_KEY"));
            assert_eq!(loaded.base_url.as_deref(), Some("https://beta.example.com"));
            assert_eq!(loaded.timeout_secs, Some(42));
            assert_eq!(loaded.family.as_deref(), Some("openai"));
            assert!(loaded
                .surfaces_json
                .as_deref()
                .unwrap_or_default()
                .contains("\"responses\":\"openai_compatible\""));

            let listed = store.get_managed_providers().await.unwrap();
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].name, "beta");

            assert!(store.delete_managed_provider("beta").await.unwrap());
            assert!(store.get_managed_provider("beta").await.unwrap().is_none());
        }

        #[tokio::test]
        async fn project_prompt_crud() {
            let store = new_store().await;

            let v1 = ProjectPromptRecord {
                project_id: "project-a".to_string(),
                prompt_name: "support".to_string(),
                version: "v1".to_string(),
                environment: "prod".to_string(),
                description: Some("Support system prompt".to_string()),
                target: "system".to_string(),
                template_text: "Support {{customer}}".to_string(),
                variables_schema_json: Some(
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "customer": { "type": "string" }
                        }
                    })
                    .to_string(),
                ),
                rollout_metadata_json: Some(serde_json::json!({"channel":"stable"}).to_string()),
                active: true,
                updated_at: "1".to_string(),
            };
            let v2 = ProjectPromptRecord {
                project_id: "project-a".to_string(),
                prompt_name: "support".to_string(),
                version: "v2".to_string(),
                environment: "prod".to_string(),
                description: Some("Support system prompt v2".to_string()),
                target: "system".to_string(),
                template_text: "Escalate {{customer}}".to_string(),
                variables_schema_json: None,
                rollout_metadata_json: None,
                active: false,
                updated_at: "2".to_string(),
            };

            store.upsert_project_prompt(&v1).await.unwrap();
            store.upsert_project_prompt(&v2).await.unwrap();

            let got = store
                .get_project_prompt("project-a", "support", "v1")
                .await
                .unwrap()
                .expect("prompt v1");
            assert_eq!(got.environment, "prod");
            assert_eq!(got.template_text, "Support {{customer}}");
            assert!(got.active);

            let prompts = store
                .get_project_prompts(Some("project-a"), Some("support"))
                .await
                .unwrap();
            assert_eq!(prompts.len(), 2);
            assert_eq!(prompts[0].version, "v1");
            assert_eq!(prompts[1].version, "v2");

            assert!(store
                .delete_project_prompt("project-a", "support", "v1")
                .await
                .unwrap());
            assert!(store
                .get_project_prompt("project-a", "support", "v1")
                .await
                .unwrap()
                .is_none());
        }

        #[tokio::test]
        async fn project_rollout_policy_crud() {
            let store = new_store().await;

            let record = ProjectRolloutPolicyRecord {
                project_id: "project-a".to_string(),
                policy_name: "prod-strict".to_string(),
                description: Some("Strict production rollout".to_string()),
                gate_config_json: serde_json::json!({
                    "preset": "strict",
                    "max_regressions": 0
                })
                .to_string(),
                target_environment: Some("prod".to_string()),
                updated_at: "1".to_string(),
            };

            store.upsert_project_rollout_policy(&record).await.unwrap();

            let got = store
                .get_project_rollout_policy("project-a", "prod-strict")
                .await
                .unwrap()
                .expect("rollout policy");
            assert_eq!(got.target_environment.as_deref(), Some("prod"));
            assert!(got.gate_config_json.contains("\"preset\":\"strict\""));

            let policies = store
                .get_project_rollout_policies(Some("project-a"))
                .await
                .unwrap();
            assert_eq!(policies.len(), 1);
            assert_eq!(policies[0].policy_name, "prod-strict");

            assert!(store
                .delete_project_rollout_policy("project-a", "prod-strict")
                .await
                .unwrap());
            assert!(store
                .get_project_rollout_policy("project-a", "prod-strict")
                .await
                .unwrap()
                .is_none());
        }

        #[tokio::test]
        async fn project_prompt_rollout_crud() {
            let store = new_store().await;

            let record = ProjectPromptRolloutRecord {
                project_id: "project-a".to_string(),
                prompt_name: "support-reset".to_string(),
                rollout_id: "rollout-1".to_string(),
                policy_name: "prod-strict".to_string(),
                baseline_version: Some("v1".to_string()),
                candidate_version: "v2".to_string(),
                baseline_run_id: "eval-1".to_string(),
                candidate_run_id: "eval-2".to_string(),
                target_environment: Some("prod".to_string()),
                status: "ready".to_string(),
                recommendation_action: Some("promote".to_string()),
                comparison_json: serde_json::json!({
                    "gate": {
                        "passed": true
                    }
                })
                .to_string(),
                created_at: "1".to_string(),
                applied_at: None,
            };

            store.upsert_project_prompt_rollout(&record).await.unwrap();

            let got = store
                .get_project_prompt_rollout("project-a", "support-reset", "rollout-1")
                .await
                .unwrap()
                .expect("prompt rollout");
            assert_eq!(got.candidate_version, "v2");
            assert_eq!(got.recommendation_action.as_deref(), Some("promote"));

            let rollouts = store
                .get_project_prompt_rollouts("project-a", "support-reset")
                .await
                .unwrap();
            assert_eq!(rollouts.len(), 1);
            assert_eq!(rollouts[0].rollout_id, "rollout-1");
        }

        #[tokio::test]
        async fn project_dataset_crud() {
            let store = new_store().await;

            let dataset = ProjectDatasetRecord {
                project_id: "project-a".to_string(),
                dataset_name: "support-replay".to_string(),
                description: Some("Support replay set".to_string()),
                schema_json: Some(
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "messages": { "type": "array" }
                        }
                    })
                    .to_string(),
                ),
                updated_at: "1".to_string(),
            };
            let item = ProjectDatasetItemRecord {
                project_id: "project-a".to_string(),
                dataset_name: "support-replay".to_string(),
                item_id: "case-1".to_string(),
                input_json: serde_json::json!({
                    "messages": [{"role":"user","content":"Help me reset my password"}]
                })
                .to_string(),
                expected_output_json: Some(
                    serde_json::json!({"contains":"reset your password"}).to_string(),
                ),
                metadata_json: Some(serde_json::json!({"priority":"high"}).to_string()),
                updated_at: "2".to_string(),
            };

            store.upsert_project_dataset(&dataset).await.unwrap();
            store.upsert_project_dataset_item(&item).await.unwrap();

            let got = store
                .get_project_dataset("project-a", "support-replay")
                .await
                .unwrap()
                .expect("dataset");
            assert_eq!(got.description.as_deref(), Some("Support replay set"));

            let datasets = store.get_project_datasets(Some("project-a")).await.unwrap();
            assert_eq!(datasets.len(), 1);
            assert_eq!(datasets[0].dataset_name, "support-replay");

            let got_item = store
                .get_project_dataset_item("project-a", "support-replay", "case-1")
                .await
                .unwrap()
                .expect("dataset item");
            assert!(got_item.input_json.contains("reset my password"));

            let items = store
                .get_project_dataset_items("project-a", "support-replay")
                .await
                .unwrap();
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].item_id, "case-1");

            assert!(store
                .delete_project_dataset_item("project-a", "support-replay", "case-1")
                .await
                .unwrap());
            assert!(store
                .get_project_dataset_item("project-a", "support-replay", "case-1")
                .await
                .unwrap()
                .is_none());

            store.upsert_project_dataset_item(&item).await.unwrap();
            assert!(store
                .delete_project_dataset("project-a", "support-replay")
                .await
                .unwrap());
            assert!(store
                .get_project_dataset("project-a", "support-replay")
                .await
                .unwrap()
                .is_none());
            assert!(store
                .get_project_dataset_item("project-a", "support-replay", "case-1")
                .await
                .unwrap()
                .is_none());
        }

        #[tokio::test]
        async fn project_eval_run_crud() {
            let store = new_store().await;

            let run = ProjectEvalRunRecord {
                run_id: "eval-1".to_string(),
                project_id: "project-a".to_string(),
                dataset_name: "support-replay".to_string(),
                target_url: "http://127.0.0.1:9999/v1/chat/completions".to_string(),
                status: "completed".to_string(),
                total_items: 1,
                passed_items: 1,
                failed_items: 0,
                total_input_tokens: 12,
                total_output_tokens: 7,
                total_cost: 0.021,
                average_latency_ms: 42.0,
                summary_json: Some(serde_json::json!({"pass_rate": 1.0}).to_string()),
                created_at: "123".to_string(),
                completed_at: Some("124".to_string()),
            };
            let item = ProjectEvalRunItemRecord {
                run_id: "eval-1".to_string(),
                project_id: "project-a".to_string(),
                dataset_name: "support-replay".to_string(),
                item_id: "case-1".to_string(),
                passed: true,
                status_code: Some(200),
                latency_ms: 42,
                output_text: Some("Reset password instructions".to_string()),
                evaluation_json: Some(
                    serde_json::json!({"kind": "expectation", "passed": true}).to_string(),
                ),
                error: None,
                input_tokens: 12,
                output_tokens: 7,
                cost: 0.021,
                created_at: "124".to_string(),
            };

            store.upsert_project_eval_run(&run).await.unwrap();
            store.upsert_project_eval_run_item(&item).await.unwrap();

            let got_run = store
                .get_project_eval_run("project-a", "eval-1")
                .await
                .unwrap()
                .expect("eval run");
            assert_eq!(got_run.dataset_name, "support-replay");
            assert_eq!(got_run.passed_items, 1);

            let all_runs = store
                .get_project_eval_runs("project-a", Some("support-replay"))
                .await
                .unwrap();
            assert_eq!(all_runs.len(), 1);
            assert_eq!(all_runs[0].run_id, "eval-1");

            let items = store
                .get_project_eval_run_items("project-a", "eval-1")
                .await
                .unwrap();
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].item_id, "case-1");
            assert_eq!(items[0].status_code, Some(200));
        }

        #[tokio::test]
        async fn governance_history_crud_and_filters() {
            let store = new_store().await;

            let changes = [
                GovernanceChangeRecord {
                    change_id: "chg-1".to_string(),
                    project_id: "project-a".to_string(),
                    resource_type: "project_tool".to_string(),
                    resource_id: "web_search".to_string(),
                    action: "upsert".to_string(),
                    before_json: None,
                    after_json: Some(
                        serde_json::json!({
                            "tool_name": "web_search",
                            "description": "Search docs v1",
                        })
                        .to_string(),
                    ),
                    changed_at: "2026-03-10T10:00:00Z".to_string(),
                },
                GovernanceChangeRecord {
                    change_id: "chg-2".to_string(),
                    project_id: "project-a".to_string(),
                    resource_type: "project_prompt".to_string(),
                    resource_id: "support:v1".to_string(),
                    action: "upsert".to_string(),
                    before_json: None,
                    after_json: Some(
                        serde_json::json!({
                            "prompt_name": "support",
                            "version": "v1",
                            "active": true,
                        })
                        .to_string(),
                    ),
                    changed_at: "2026-03-10T10:00:01Z".to_string(),
                },
                GovernanceChangeRecord {
                    change_id: "chg-3".to_string(),
                    project_id: "project-a".to_string(),
                    resource_type: "project_tool".to_string(),
                    resource_id: "web_search".to_string(),
                    action: "delete".to_string(),
                    before_json: Some(
                        serde_json::json!({
                            "tool_name": "web_search",
                            "description": "Search docs v1",
                        })
                        .to_string(),
                    ),
                    after_json: None,
                    changed_at: "2026-03-10T10:00:02Z".to_string(),
                },
            ];

            for change in &changes {
                store.append_governance_change(change).await.unwrap();
            }

            let all = store
                .get_governance_changes("project-a", None, 100)
                .await
                .unwrap();
            assert_eq!(all.len(), 3);
            assert_eq!(all[0].change_id, "chg-3");
            assert_eq!(all[1].change_id, "chg-2");
            assert_eq!(all[2].change_id, "chg-1");

            let tool_only = store
                .get_governance_changes("project-a", Some("project_tool"), 100)
                .await
                .unwrap();
            assert_eq!(tool_only.len(), 2);
            assert!(tool_only
                .iter()
                .all(|change| change.resource_type == "project_tool"));

            let limited = store
                .get_governance_changes("project-a", None, 1)
                .await
                .unwrap();
            assert_eq!(limited.len(), 1);
            assert_eq!(limited[0].change_id, "chg-3");
        }
    }

    // -----------------------------------------------------------------------
    // CostTracker store integration tests
    // -----------------------------------------------------------------------

    mod cost_tracker_store {
        use super::*;

        fn tracker_config() -> toml::Value {
            toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(10.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t.insert("default_cost_per_1k_input".into(), toml::Value::Float(0.01));
                t.insert(
                    "default_cost_per_1k_output".into(),
                    toml::Value::Float(0.02),
                );
                t
            })
        }

        #[tokio::test]
        async fn flush_writes_to_store() {
            let s = store::connect("sqlite::memory:").await.unwrap();
            let s = Arc::new(s);

            let tracker = cost_tracker::create_tracker(&tracker_config())
                .unwrap()
                .with_store(Arc::clone(&s));

            // Insert some usage into DashMap.
            tracker.set_model_cost("gpt-4", 0.03, 0.06);

            // Flush.
            tracker.flush_to_store().await.unwrap();

            // Verify in the store.
            let models = s.get_all_model_costs().await.unwrap();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].0, "gpt-4");
            assert!((models[0].1.input_cost_per_1k - 0.03).abs() < 1e-9);
        }

        #[tokio::test]
        async fn load_restores_from_store() {
            let s = store::connect("sqlite::memory:").await.unwrap();
            let s = Arc::new(s);

            // Pre-populate the store.
            s.upsert_usage(
                "sk-restored",
                &KeyUsageRecord {
                    total_input_tokens: 500,
                    total_output_tokens: 250,
                    total_cost: 0.025,
                },
            )
            .await
            .unwrap();

            s.upsert_model_cost(
                "claude-3",
                &ModelCostRecord {
                    input_cost_per_1k: 0.015,
                    output_cost_per_1k: 0.075,
                },
            )
            .await
            .unwrap();

            // Create tracker and load.
            let tracker = cost_tracker::create_tracker(&tracker_config())
                .unwrap()
                .with_store(Arc::clone(&s));
            tracker.load_from_store().await.unwrap();

            // Verify loaded.
            let usage = tracker.get_usage("sk-restored").unwrap();
            assert_eq!(usage.total_input_tokens, 500);
            assert!((usage.total_cost - 0.025).abs() < 1e-9);

            let models = tracker.get_model_costs();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].0, "claude-3");
        }
    }

    // -----------------------------------------------------------------------
    // LlmGatewayApi tests
    // -----------------------------------------------------------------------

    mod api_tests {
        use super::*;

        fn tracker_config() -> toml::Value {
            toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(10.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t
            })
        }

        fn limiter_config() -> toml::Value {
            toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("tokens_per_minute".into(), toml::Value::Float(60_000.0));
                t.insert("burst_tokens".into(), toml::Value::Float(10_000.0));
                t
            })
        }

        fn failover_config() -> toml::Value {
            toml::Value::Table({
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
            })
        }

        #[tokio::test]
        async fn api_reports_plugin_status() {
            let ct = cost_tracker::create_tracker(&tracker_config()).unwrap();
            let rl = rate_limiter::create_limiter(&limiter_config()).unwrap();
            let pf = provider_failover::create_failover(&failover_config()).unwrap();

            let api = LlmGatewayApi::new(
                Some(ct),
                Some(rl),
                Some(pf),
                None,
                None,
                None,
                None,
                None,
                None,
            );

            assert!(api.cost_tracker_enabled());
            assert!(api.rate_limiter_enabled());
            assert!(api.provider_failover_enabled());
        }

        #[tokio::test]
        async fn api_disabled_plugins_return_none() {
            let api = LlmGatewayApi::new(None, None, None, None, None, None, None, None, None);

            assert!(!api.cost_tracker_enabled());
            assert!(api.cost_usage().is_none());
            assert!(api.budget_limit().is_none());
            assert!(api.model_costs().is_none());
            assert!(api.rate_limiter_tracked_keys().is_none());
            assert!(api.rate_limiter_config().is_none());
            assert!(api.providers().is_none());
            assert!(api.failed_providers().is_none());
        }

        #[tokio::test]
        async fn api_cost_mutations() {
            let ct = cost_tracker::create_tracker(&tracker_config()).unwrap();
            let api = LlmGatewayApi::new(Some(ct), None, None, None, None, None, None, None, None);

            // Set model cost.
            api.set_model_cost("gpt-4", 0.03, 0.06).await;
            let models = api.model_costs().unwrap();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].0, "gpt-4");

            // Delete model cost.
            assert_eq!(api.delete_model_cost("gpt-4").await, Some(true));
            assert_eq!(api.delete_model_cost("gpt-4").await, Some(false));
            assert_eq!(api.model_costs().unwrap().len(), 0);
        }

        #[tokio::test]
        async fn api_provider_management() {
            let pf = provider_failover::create_failover(&failover_config()).unwrap();
            let api = LlmGatewayApi::new(
                None,
                None,
                Some(pf.clone()),
                None,
                None,
                None,
                None,
                None,
                None,
            );

            // Initially no failures.
            assert_eq!(api.failed_providers().unwrap().len(), 0);

            // We can only test clearing, since marking failures requires the on_error hook.
            // Verify providers are listed.
            let providers = api.providers().unwrap();
            assert_eq!(providers.len(), 1);
            assert_eq!(providers[0].name, "openai");

            // Clear all (no-op when empty).
            api.clear_all_failed_providers();
            assert_eq!(api.failed_providers().unwrap().len(), 0);
        }

        #[tokio::test]
        async fn api_rate_limiter_info() {
            let rl = rate_limiter::create_limiter(&limiter_config()).unwrap();
            let api = LlmGatewayApi::new(None, Some(rl), None, None, None, None, None, None, None);

            let (rate, burst) = api.rate_limiter_config().unwrap();
            assert!((rate - 1000.0).abs() < 1e-3); // 60000/60 = 1000
            assert!((burst - 10000.0).abs() < 1e-3);
            assert_eq!(api.rate_limiter_tracked_keys().unwrap(), 0);
        }
    }

    // -----------------------------------------------------------------------
    // create_plugins() integration test
    // -----------------------------------------------------------------------

    mod create_plugins_tests {
        use proxy_core::config::PluginConfig;

        #[tokio::test]
        async fn creates_all_plugins_with_sqlite_store() {
            let configs = vec![
                PluginConfig {
                    name: "cost_tracker".into(),
                    enabled: true,
                    config: toml::Value::Table({
                        let mut t = toml::value::Map::new();
                        t.insert("budget_limit".into(), toml::Value::Float(10.0));
                        t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                        t
                    }),
                },
                PluginConfig {
                    name: "rate_limiter".into(),
                    enabled: true,
                    config: toml::Value::Table({
                        let mut t = toml::value::Map::new();
                        t.insert("tokens_per_minute".into(), toml::Value::Float(60_000.0));
                        t.insert("burst_tokens".into(), toml::Value::Float(10_000.0));
                        t
                    }),
                },
                PluginConfig {
                    name: "provider_failover".into(),
                    enabled: true,
                    config: toml::Value::Table(toml::value::Map::new()),
                },
            ];

            let (plugins, api) = plugin_llm_gateway::create_plugins(
                &configs,
                Some("sqlite::memory:"),
                &[],
                &[],
                None,
            )
            .await
            .unwrap();

            assert_eq!(plugins.len(), 3);
            assert!(api.cost_tracker_enabled());
            assert!(api.rate_limiter_enabled());
            assert!(api.provider_failover_enabled());
        }

        #[tokio::test]
        async fn creates_plugins_without_store() {
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

            let (plugins, api) = plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None)
                .await
                .unwrap();

            assert_eq!(plugins.len(), 1);
            assert!(api.cost_tracker_enabled());
        }

        #[tokio::test]
        async fn skips_disabled_plugins() {
            let configs = vec![PluginConfig {
                name: "cost_tracker".into(),
                enabled: false,
                config: toml::Value::Table(toml::value::Map::new()),
            }];

            let (plugins, api) = plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None)
                .await
                .unwrap();

            assert_eq!(plugins.len(), 0);
            assert!(!api.cost_tracker_enabled());
        }

        #[tokio::test]
        async fn semantic_safety_requires_safe_config_order() {
            let semantic_config = PluginConfig {
                name: "semantic_safety".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert(
                        "endpoint".into(),
                        toml::Value::String("http://127.0.0.1:50061".into()),
                    );
                    t
                }),
            };
            let content_filter_config = PluginConfig {
                name: "content_filter".into(),
                enabled: true,
                config: toml::Value::Table(toml::value::Map::new()),
            };

            let configs = vec![content_filter_config.clone(), semantic_config.clone()];
            let (plugins, _) = plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None)
                .await
                .unwrap();

            let names = plugins
                .iter()
                .map(|plugin| plugin.name().to_string())
                .collect::<Vec<_>>();
            assert_eq!(names, vec!["content_filter", "semantic_safety"]);

            let configs = vec![semantic_config, content_filter_config];
            let error =
                match plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None).await {
                    Ok(_) => panic!("unsafe semantic_safety ordering should fail at startup"),
                    Err(error) => error,
                };
            assert!(
                error
                    .to_string()
                    .contains("semantic_safety must be configured after content_filter"),
                "unexpected error: {error}"
            );
        }

        #[tokio::test]
        async fn prompt_registry_requires_safe_config_order() {
            let prompt_registry_config = PluginConfig {
                name: "prompt_registry".into(),
                enabled: true,
                config: toml::Value::Table(toml::value::Map::new()),
            };
            let prompt_cache_config = PluginConfig {
                name: "prompt_cache".into(),
                enabled: true,
                config: toml::Value::Table(toml::value::Map::new()),
            };

            let configs = vec![prompt_registry_config.clone(), prompt_cache_config.clone()];
            let (plugins, _) = plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None)
                .await
                .unwrap();

            let names = plugins
                .iter()
                .map(|plugin| plugin.name().to_string())
                .collect::<Vec<_>>();
            assert_eq!(names, vec!["prompt_registry", "prompt_cache"]);

            let configs = vec![prompt_cache_config, prompt_registry_config];
            let error =
                match plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None).await {
                    Ok(_) => panic!("unsafe prompt_registry ordering should fail at startup"),
                    Err(error) => error,
                };
            assert!(
                error
                    .to_string()
                    .contains("prompt_registry must be configured before prompt_cache"),
                "unexpected error: {error}"
            );
        }

        #[tokio::test]
        async fn semantic_cache_requires_safe_config_order() {
            let content_filter_config = PluginConfig {
                name: "content_filter".into(),
                enabled: true,
                config: toml::Value::Table(toml::value::Map::new()),
            };
            let semantic_cache_config = PluginConfig {
                name: "semantic_cache".into(),
                enabled: true,
                config: toml::Value::Table(toml::value::Map::new()),
            };
            let prompt_cache_config = PluginConfig {
                name: "prompt_cache".into(),
                enabled: true,
                config: toml::Value::Table(toml::value::Map::new()),
            };

            let configs = vec![
                content_filter_config.clone(),
                semantic_cache_config.clone(),
                prompt_cache_config.clone(),
            ];
            let (plugins, _) = plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None)
                .await
                .unwrap();
            let names = plugins
                .iter()
                .map(|plugin| plugin.name().to_string())
                .collect::<Vec<_>>();
            assert_eq!(
                names,
                vec!["content_filter", "semantic_cache", "prompt_cache"]
            );

            let configs = vec![semantic_cache_config.clone(), content_filter_config];
            let error =
                match plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None).await {
                    Ok(_) => panic!("unsafe semantic_cache ordering should fail at startup"),
                    Err(error) => error,
                };
            assert!(
                error
                    .to_string()
                    .contains("semantic_cache must be configured after content_filter"),
                "unexpected error: {error}"
            );

            let configs = vec![prompt_cache_config, semantic_cache_config];
            let error =
                match plugin_llm_gateway::create_plugins(&configs, None, &[], &[], None).await {
                    Ok(_) => panic!("semantic_cache after prompt_cache should fail at startup"),
                    Err(error) => error,
                };
            assert!(
                error
                    .to_string()
                    .contains("semantic_cache must be configured before prompt_cache"),
                "unexpected error: {error}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Management server integration tests
    // -----------------------------------------------------------------------

    mod management_server_tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;
        use std::time::Duration;

        use bytes::Bytes;
        use http_body_util::{BodyExt, Full};
        use hyper_util::client::legacy::Client;
        use hyper_util::rt::TokioExecutor;
        use plugin_llm_gateway::CreatePluginsOptions;
        use serde_json::Value;
        use tempfile::NamedTempFile;

        async fn start_test_server(api: LlmGatewayApi) -> u16 {
            // Bind to port 0 to get an ephemeral port.
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();

            // Start the management server on this listener manually.
            tokio::spawn(async move {
                use hyper::body::Incoming;
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
                        let svc = service_fn(move |req: hyper::Request<Incoming>| {
                            let api = api.clone();
                            async move {
                                plugin_llm_gateway::management_server::handle_request(req, api)
                                    .await
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

        async fn get(port: u16, path: &str) -> (u16, String) {
            let client = Client::builder(TokioExecutor::new()).build_http::<Full<Bytes>>();
            let uri = format!("http://127.0.0.1:{}{}", port, path);
            let resp = client.get(uri.parse().unwrap()).await.unwrap();
            let status = resp.status().as_u16();
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(body.to_vec()).unwrap())
        }

        async fn delete(port: u16, path: &str) -> (u16, String) {
            let client = Client::builder(TokioExecutor::new()).build_http::<Full<Bytes>>();
            let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
                .parse()
                .unwrap();
            let req = hyper::Request::builder()
                .method("DELETE")
                .uri(uri)
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = client.request(req).await.unwrap();
            let status = resp.status().as_u16();
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(body.to_vec()).unwrap())
        }

        async fn put(port: u16, path: &str, body: &str) -> (u16, String) {
            let client = Client::builder(TokioExecutor::new()).build_http::<Full<Bytes>>();
            let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
                .parse()
                .unwrap();
            let req = hyper::Request::builder()
                .method("PUT")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(body.to_string())))
                .unwrap();
            let resp = client.request(req).await.unwrap();
            let status = resp.status().as_u16();
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(body.to_vec()).unwrap())
        }

        async fn put_with_bearer(port: u16, path: &str, body: &str, token: &str) -> (u16, String) {
            let client = Client::builder(TokioExecutor::new()).build_http::<Full<Bytes>>();
            let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
                .parse()
                .unwrap();
            let req = hyper::Request::builder()
                .method("PUT")
                .uri(uri)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token))
                .body(Full::new(Bytes::from(body.to_string())))
                .unwrap();
            let resp = client.request(req).await.unwrap();
            let status = resp.status().as_u16();
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(body.to_vec()).unwrap())
        }

        async fn post_with_bearer(port: u16, path: &str, body: &str, token: &str) -> (u16, String) {
            let client = Client::builder(TokioExecutor::new()).build_http::<Full<Bytes>>();
            let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
                .parse()
                .unwrap();
            let req = hyper::Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token))
                .body(Full::new(Bytes::from(body.to_string())))
                .unwrap();
            let resp = client.request(req).await.unwrap();
            let status = resp.status().as_u16();
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(body.to_vec()).unwrap())
        }

        async fn get_with_bearer(port: u16, path: &str, token: &str) -> (u16, String) {
            let client = Client::builder(TokioExecutor::new()).build_http::<Full<Bytes>>();
            let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
                .parse()
                .unwrap();
            let req = hyper::Request::builder()
                .method("GET")
                .uri(uri)
                .header("authorization", format!("Bearer {}", token))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = client.request(req).await.unwrap();
            let status = resp.status().as_u16();
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(body.to_vec()).unwrap())
        }

        async fn delete_with_bearer(port: u16, path: &str, token: &str) -> (u16, String) {
            let client = Client::builder(TokioExecutor::new()).build_http::<Full<Bytes>>();
            let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
                .parse()
                .unwrap();
            let req = hyper::Request::builder()
                .method("DELETE")
                .uri(uri)
                .header("authorization", format!("Bearer {}", token))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = client.request(req).await.unwrap();
            let status = resp.status().as_u16();
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8(body.to_vec()).unwrap())
        }

        async fn start_static_eval_target_server(content: &str) -> u16 {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let content = content.to_string();
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    let content = content.clone();
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| {
                            let content = content.clone();
                            async move {
                                let _body = req.into_body().collect().await.unwrap().to_bytes();
                                let response = serde_json::json!({
                                    "choices": [{
                                        "message": {
                                            "role": "assistant",
                                            "content": content,
                                        }
                                    }],
                                    "usage": {
                                        "prompt_tokens": 10,
                                        "completion_tokens": 8,
                                        "total_cost": 0.004
                                    }
                                });
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(200)
                                        .header("content-type", "application/json")
                                        .body(Full::new(Bytes::from(response.to_string())))
                                        .unwrap(),
                                )
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

        async fn seed_rollout_eval_dataset(port: u16, mgmt_token: &str, dataset_name: &str) {
            let dataset_payload = serde_json::json!({
                "description": "Canary rollout evaluation dataset",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/datasets/{dataset_name}"),
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "contains": ["reset password"],
                    "status_code": 200
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/datasets/{dataset_name}/items/case-1"),
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");
        }

        async fn execute_prompt_eval_run(
            port: u16,
            mgmt_token: &str,
            dataset_name: &str,
            target_port: u16,
            prompt_version: &str,
        ) -> String {
            let eval_payload = serde_json::json!({
                "dataset_name": dataset_name,
                "target_url": format!("http://127.0.0.1:{target_port}/v1/chat/completions"),
                "timeout_ms": 1000,
                "provider_name": "openai",
                "model": "gpt-4o-mini",
                "prompt_name": "support-reset",
                "prompt_version": prompt_version,
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "eval run execution failed: {body}");
            serde_json::from_str::<Value>(&body).unwrap()["run"]["run_id"]
                .as_str()
                .expect("run id present")
                .to_string()
        }

        async fn seed_prompt_versions(api: &LlmGatewayApi) {
            for (version, active) in [("v1", true), ("v2", false)] {
                api.upsert_project_prompt(ProjectPromptRecord {
                    project_id: "project-a".to_string(),
                    prompt_name: "support-reset".to_string(),
                    version: version.to_string(),
                    environment: "prod".to_string(),
                    description: Some(format!("Prompt {version}")),
                    target: "system".to_string(),
                    template_text: format!("Prompt version {version}"),
                    variables_schema_json: None,
                    rollout_metadata_json: None,
                    active,
                    updated_at: current_timestamp_string(),
                })
                .await
                .expect("governance enabled")
                .expect("upsert prompt");
            }
        }

        async fn seed_ready_prompt_rollout(
            api: &LlmGatewayApi,
            rollout_id: &str,
            policy_name: &str,
        ) {
            api.upsert_project_prompt_rollout(ProjectPromptRolloutRecord {
                project_id: "project-a".to_string(),
                prompt_name: "support-reset".to_string(),
                rollout_id: rollout_id.to_string(),
                policy_name: policy_name.to_string(),
                baseline_version: Some("v1".to_string()),
                candidate_version: "v2".to_string(),
                baseline_run_id: "run-base".to_string(),
                candidate_run_id: "run-candidate".to_string(),
                target_environment: Some("prod".to_string()),
                status: "ready".to_string(),
                recommendation_action: Some("canary".to_string()),
                comparison_json: serde_json::json!({
                    "summary": {
                        "candidate_pass_rate": 1.0
                    }
                })
                .to_string(),
                created_at: current_timestamp_string(),
                applied_at: None,
            })
            .await
            .expect("governance enabled")
            .expect("upsert rollout");
        }

        #[tokio::test]
        async fn status_endpoint() {
            let ct = cost_tracker::create_tracker(&toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(10.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t
            }))
            .unwrap();
            let api = LlmGatewayApi::new(Some(ct), None, None, None, None, None, None, None, None);

            let port = start_test_server(api).await;
            let (status, body) = get(port, "/api/v1/status").await;
            assert_eq!(status, 200);
            assert!(body.contains("\"cost_tracker_enabled\":true"));
            assert!(body.contains("\"rate_limiter_enabled\":false"));
        }

        #[tokio::test]
        async fn cost_usage_endpoint() {
            let ct = cost_tracker::create_tracker(&toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(5.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t
            }))
            .unwrap();
            let api = LlmGatewayApi::new(Some(ct), None, None, None, None, None, None, None, None);

            let port = start_test_server(api).await;

            let (status, body) = get(port, "/api/v1/cost/usage").await;
            assert_eq!(status, 200);
            assert!(body.contains("\"budget_limit\":5.000000"));
            assert!(body.contains("\"usage\":[]"));
        }

        #[tokio::test]
        async fn model_cost_put_and_delete() {
            let ct = cost_tracker::create_tracker(&toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(0.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t
            }))
            .unwrap();
            let api = LlmGatewayApi::new(Some(ct), None, None, None, None, None, None, None, None);

            let port = start_test_server(api).await;

            // PUT a model cost.
            let (status, body) = put(
                port,
                "/api/v1/cost/models/gpt-4",
                r#"{"input_cost_per_1k":0.03,"output_cost_per_1k":0.06}"#,
            )
            .await;
            assert_eq!(status, 200);
            assert!(body.contains("\"ok\":true"));

            // GET models to verify.
            let (status, body) = get(port, "/api/v1/cost/models").await;
            assert_eq!(status, 200);
            assert!(body.contains("gpt-4"));
            assert!(body.contains("0.030000"));

            // DELETE model cost.
            let (status, body) = delete(port, "/api/v1/cost/models/gpt-4").await;
            assert_eq!(status, 200);
            assert!(body.contains("\"deleted\":true"));

            // DELETE again - should return 404.
            let (status, _) = delete(port, "/api/v1/cost/models/gpt-4").await;
            assert_eq!(status, 404);
        }

        #[tokio::test]
        async fn not_found_for_unknown_path() {
            let api = LlmGatewayApi::new(None, None, None, None, None, None, None, None, None);
            let port = start_test_server(api).await;

            let (status, body) = get(port, "/unknown").await;
            assert_eq!(status, 404);
            assert!(body.contains("not found"));
        }

        #[tokio::test]
        async fn rate_limiter_status_endpoint() {
            let rl = rate_limiter::create_limiter(&toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("tokens_per_minute".into(), toml::Value::Float(60_000.0));
                t.insert("burst_tokens".into(), toml::Value::Float(10_000.0));
                t
            }))
            .unwrap();
            let api = LlmGatewayApi::new(None, Some(rl), None, None, None, None, None, None, None);

            let port = start_test_server(api).await;
            let (status, body) = get(port, "/api/v1/rate-limiter/status").await;
            assert_eq!(status, 200);
            assert!(body.contains("\"rate_per_second\":1000.00"));
            assert!(body.contains("\"burst\":10000"));
        }

        #[tokio::test]
        async fn providers_endpoints() {
            let pf = provider_failover::create_failover(&toml::Value::Table({
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
            }))
            .unwrap();
            let api = LlmGatewayApi::new(None, None, Some(pf), None, None, None, None, None, None);

            let port = start_test_server(api).await;

            // GET providers.
            let (status, body) = get(port, "/api/v1/providers").await;
            assert_eq!(status, 200);
            assert!(body.contains("\"name\":\"openai\""));
            assert!(body.contains("\"cooldown_secs\":30"));

            // GET failed - should be empty.
            let (status, body) = get(port, "/api/v1/providers/failed").await;
            assert_eq!(status, 200);
            assert!(body.contains("\"failed\":[]"));
        }

        #[tokio::test]
        async fn semantic_cache_status_endpoint() {
            let semantic_cache =
                plugin_llm_gateway::semantic_cache::create_plugin(&toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("default_ttl_secs".into(), toml::Value::Integer(600));
                    t.insert(
                        "default_similarity_threshold".into(),
                        toml::Value::Float(0.9),
                    );
                    t.insert("max_entries".into(), toml::Value::Integer(64));
                    t
                }))
                .unwrap();
            let api = LlmGatewayApi::new(
                None,
                None,
                None,
                None,
                None,
                Some(semantic_cache),
                None,
                None,
                None,
            );

            let port = start_test_server(api).await;
            let (status, body) = get(port, "/api/v1/semantic-cache/status").await;
            assert_eq!(status, 200);
            assert!(body.contains("\"default_ttl_secs\":600"));
            assert!(body.contains("\"default_similarity_threshold\":0.9"));
            assert!(body.contains("\"store_backed\":false"));
            assert!(body.contains("\"entry_count\":0"));
        }

        #[tokio::test]
        async fn semantic_policy_put_generates_unique_versions_when_omitted() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let (status, first_body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/semantic-safety",
                r#"{"enabled":true,"entities":[],"topics":[]}"#,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let first_json: Value = serde_json::from_str(&first_body).unwrap();
            let first_version = first_json
                .get("policy_version")
                .and_then(|value| value.as_str())
                .unwrap()
                .to_string();

            let (status, second_body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/semantic-safety",
                r#"{"enabled":true,"entities":[],"topics":[]}"#,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let second_json: Value = serde_json::from_str(&second_body).unwrap();
            let second_version = second_json
                .get("policy_version")
                .and_then(|value| value.as_str())
                .unwrap()
                .to_string();

            assert_ne!(first_version, second_version);
            assert!(first_version.starts_with("sem-"));
            assert!(second_version.starts_with("sem-"));
        }

        #[tokio::test]
        async fn semantic_policy_put_preserves_explicit_version() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/semantic-safety",
                r#"{"version":"caller-v1","enabled":true,"entities":[],"topics":[]}"#,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let response_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                response_json
                    .get("policy_version")
                    .and_then(|value| value.as_str()),
                Some("caller-v1")
            );

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/semantic-safety",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let policy_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                policy_json.get("version").and_then(|value| value.as_str()),
                Some("caller-v1")
            );
        }

        #[tokio::test]
        async fn project_tools_management_round_trip() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let payload = serde_json::json!({
                "description": "Search docs",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                },
                "executor_kind": "webhook",
                "executor_config": {
                    "url": "http://tool.local/search",
                    "method": "POST"
                },
                "enabled": true,
                "timeout_ms": 1500
            })
            .to_string();

            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/tools/web_search",
                &payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "tool upsert failed: {body}");

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/tools/web_search",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let tool_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                tool_json.get("tool_name").and_then(|value| value.as_str()),
                Some("web_search")
            );
            assert_eq!(
                tool_json
                    .get("executor_kind")
                    .and_then(|value| value.as_str()),
                Some("webhook")
            );
            assert_eq!(
                tool_json
                    .get("executor_config")
                    .and_then(|value| value.get("url"))
                    .and_then(|value| value.as_str()),
                Some("http://tool.local/search")
            );

            let (status, body) =
                get_with_bearer(port, "/api/v1/projects/project-a/tools", mgmt_token).await;
            assert_eq!(status, 200);
            assert!(body.contains("\"tool_name\":\"web_search\""));

            let (status, _) = delete_with_bearer(
                port,
                "/api/v1/projects/project-a/tools/web_search",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);

            let (status, _) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/tools/web_search",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 404);
        }

        #[tokio::test]
        async fn project_prompts_management_round_trip_and_active_switch() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let v1_payload = serde_json::json!({
                "environment": "prod",
                "description": "Primary support prompt",
                "target": "system",
                "template_text": "You are helping {{customer}}.",
                "variables_schema": {
                    "type": "object",
                    "properties": {
                        "customer": { "type": "string" }
                    },
                    "required": ["customer"]
                },
                "rollout_metadata": {
                    "channel": "stable"
                },
                "active": true
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support/versions/v1",
                &v1_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt v1 upsert failed: {body}");

            let v2_payload = serde_json::json!({
                "environment": "prod",
                "description": "Secondary support prompt",
                "target": "system",
                "template_text": "Escalate {{customer}}.",
                "active": true
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support/versions/v2",
                &v2_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt v2 upsert failed: {body}");

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support/versions/v1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let prompt_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(prompt_json["prompt_name"].as_str(), Some("support"));
            assert_eq!(prompt_json["version"].as_str(), Some("v1"));
            assert_eq!(prompt_json["environment"].as_str(), Some("prod"));
            assert_eq!(prompt_json["active"].as_bool(), Some(false));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let versions_json: Value = serde_json::from_str(&body).unwrap();
            let versions = versions_json["versions"]
                .as_array()
                .expect("versions array");
            assert_eq!(versions.len(), 2);
            assert_eq!(
                versions
                    .iter()
                    .find(|prompt| prompt["version"].as_str() == Some("v2"))
                    .and_then(|prompt| prompt["active"].as_bool()),
                Some(true)
            );

            let (status, body) =
                get_with_bearer(port, "/api/v1/projects/project-a/prompts", mgmt_token).await;
            assert_eq!(status, 200);
            assert!(body.contains("\"prompt_name\":\"support\""));

            let (status, _) = delete_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support/versions/v1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);

            let (status, _) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support/versions/v1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 404);
        }

        #[tokio::test]
        async fn project_rollout_policy_can_drive_prompt_promotion() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let upstream_port = upstream_listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match upstream_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| async move {
                            let _ = req.into_body().collect().await.unwrap().to_bytes();
                            let response = serde_json::json!({
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": "Use reset link 1234."
                                        }
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 9,
                                    "completion_tokens": 4,
                                    "total_cost": 0.008
                                }
                            });
                            Ok::<_, hyper::Error>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(response.to_string())))
                                    .unwrap(),
                            )
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let dataset_payload = serde_json::json!({
                "description": "Replay set for rollout promotion",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-rollout",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "contains": "reset link",
                    "status_code": 200
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-rollout/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            for (version, active) in [("v1", true), ("v2", false)] {
                let prompt_payload = serde_json::json!({
                    "environment": "prod",
                    "description": format!("Prompt {version}"),
                    "target": "system",
                    "template_text": format!("Prompt version {version}"),
                    "active": active
                })
                .to_string();
                let (status, body) = put_with_bearer(
                    port,
                    &format!("/api/v1/projects/project-a/prompts/support-reset/versions/{version}"),
                    &prompt_payload,
                    mgmt_token,
                )
                .await;
                assert_eq!(status, 200, "prompt {version} upsert failed: {body}");
            }

            let rollout_policy_payload = serde_json::json!({
                "description": "Strict prod policy",
                "gate": {
                    "preset": "strict"
                },
                "target_environment": "prod"
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/rollout-policies/prod-strict",
                &rollout_policy_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "rollout policy upsert failed: {body}");

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/rollout-policies/prod-strict",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "rollout policy fetch failed: {body}");
            let rollout_policy_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                rollout_policy_json["policy_name"].as_str(),
                Some("prod-strict")
            );
            assert_eq!(
                rollout_policy_json["gate"]["preset"].as_str(),
                Some("strict")
            );

            let target_url = format!("http://127.0.0.1:{upstream_port}/v1/chat/completions");
            let baseline_eval_payload = serde_json::json!({
                "dataset_name": "support-rollout",
                "target_url": target_url,
                "timeout_ms": 1000,
                "prompt_name": "support-reset",
                "prompt_version": "v1",
                "provider_name": "openai",
                "model": "gpt-4o-mini",
                "route_path": "/v1/chat/completions",
                "safety_profile": "standard"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &baseline_eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "baseline eval failed: {body}");
            let baseline_json: Value = serde_json::from_str(&body).unwrap();
            let baseline_run_id = baseline_json["run"]["run_id"].as_str().unwrap().to_string();

            let candidate_eval_payload = serde_json::json!({
                "dataset_name": "support-rollout",
                "target_url": format!("http://127.0.0.1:{upstream_port}/v1/chat/completions"),
                "timeout_ms": 1000,
                "prompt_name": "support-reset",
                "prompt_version": "v2",
                "provider_name": "openai",
                "model": "gpt-4o-mini",
                "route_path": "/v1/chat/completions",
                "safety_profile": "standard"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &candidate_eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "candidate eval failed: {body}");
            let candidate_json: Value = serde_json::from_str(&body).unwrap();
            let candidate_run_id = candidate_json["run"]["run_id"]
                .as_str()
                .unwrap()
                .to_string();

            let (status, body) = get_with_bearer(
                port,
                &format!(
                    "/api/v1/projects/project-a/eval-runs/compare?baseline_run_id={baseline_run_id}&candidate_run_id={candidate_run_id}&policy_name=prod-strict"
                ),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "policy comparison failed: {body}");
            let comparison_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                comparison_json["rollout_policy"]["policy_name"].as_str(),
                Some("prod-strict")
            );
            assert_eq!(comparison_json["gate"]["passed"].as_bool(), Some(true));
            assert_eq!(
                comparison_json["gate"]["recommendation"]["action"].as_str(),
                Some("promote")
            );

            let promote_payload = serde_json::json!({
                "candidate_version": "v2",
                "baseline_run_id": baseline_run_id,
                "candidate_run_id": candidate_run_id,
                "policy_name": "prod-strict"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/promote",
                &promote_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt promotion failed: {body}");
            let promote_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(promote_json["promoted"].as_bool(), Some(true));
            assert_eq!(promote_json["prompt"]["version"].as_str(), Some("v2"));
            assert_eq!(promote_json["prompt"]["active"].as_bool(), Some(true));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v1_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v1_json["active"].as_bool(), Some(false));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v2",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v2_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v2_json["active"].as_bool(), Some(true));
        }

        #[tokio::test]
        async fn project_prompt_rollouts_can_be_created_and_applied() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let upstream_port = upstream_listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match upstream_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| async move {
                            let _ = req.into_body().collect().await.unwrap().to_bytes();
                            let response = serde_json::json!({
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": "Use reset link 1234."
                                        }
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 9,
                                    "completion_tokens": 4,
                                    "total_cost": 0.008
                                }
                            });
                            Ok::<_, hyper::Error>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(response.to_string())))
                                    .unwrap(),
                            )
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let dataset_payload = serde_json::json!({
                "description": "Replay set for rollout workflow",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-rollout",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "contains": "reset link",
                    "status_code": 200
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-rollout/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            for (version, active) in [("v1", true), ("v2", false)] {
                let prompt_payload = serde_json::json!({
                    "environment": "prod",
                    "description": format!("Prompt {version}"),
                    "target": "system",
                    "template_text": format!("Prompt version {version}"),
                    "active": active
                })
                .to_string();
                let (status, body) = put_with_bearer(
                    port,
                    &format!("/api/v1/projects/project-a/prompts/support-reset/versions/{version}"),
                    &prompt_payload,
                    mgmt_token,
                )
                .await;
                assert_eq!(status, 200, "prompt {version} upsert failed: {body}");
            }

            let rollout_policy_payload = serde_json::json!({
                "description": "Strict prod policy",
                "gate": {
                    "preset": "strict"
                },
                "target_environment": "prod"
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/rollout-policies/prod-strict",
                &rollout_policy_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "rollout policy upsert failed: {body}");

            let target_url = format!("http://127.0.0.1:{upstream_port}/v1/chat/completions");
            let baseline_eval_payload = serde_json::json!({
                "dataset_name": "support-rollout",
                "target_url": target_url,
                "timeout_ms": 1000,
                "prompt_name": "support-reset",
                "prompt_version": "v1",
                "provider_name": "openai",
                "model": "gpt-4o-mini",
                "route_path": "/v1/chat/completions",
                "safety_profile": "standard"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &baseline_eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "baseline eval failed: {body}");
            let baseline_json: Value = serde_json::from_str(&body).unwrap();
            let baseline_run_id = baseline_json["run"]["run_id"].as_str().unwrap().to_string();

            let candidate_eval_payload = serde_json::json!({
                "dataset_name": "support-rollout",
                "target_url": format!("http://127.0.0.1:{upstream_port}/v1/chat/completions"),
                "timeout_ms": 1000,
                "prompt_name": "support-reset",
                "prompt_version": "v2",
                "provider_name": "openai",
                "model": "gpt-4o-mini",
                "route_path": "/v1/chat/completions",
                "safety_profile": "standard"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &candidate_eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "candidate eval failed: {body}");
            let candidate_json: Value = serde_json::from_str(&body).unwrap();
            let candidate_run_id = candidate_json["run"]["run_id"]
                .as_str()
                .unwrap()
                .to_string();

            let rollout_payload = serde_json::json!({
                "candidate_version": "v2",
                "baseline_run_id": baseline_run_id,
                "candidate_run_id": candidate_run_id,
                "policy_name": "prod-strict"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts",
                &rollout_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 201, "prompt rollout create failed: {body}");
            let rollout_json: Value = serde_json::from_str(&body).unwrap();
            let rollout_id = rollout_json["rollout"]["rollout_id"]
                .as_str()
                .unwrap()
                .to_string();
            assert_eq!(rollout_json["rollout"]["status"].as_str(), Some("ready"));
            assert_eq!(
                rollout_json["rollout"]["recommendation_action"].as_str(),
                Some("promote")
            );

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v2",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let before_apply_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(before_apply_json["active"].as_bool(), Some(false));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout list failed: {body}");
            let list_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(list_json["rollouts"].as_array().unwrap().len(), 1);

            let (status, body) = get_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/prompts/support-reset/rollouts/{rollout_id}"),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout fetch failed: {body}");
            let fetched_rollout_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                fetched_rollout_json["candidate_version"].as_str(),
                Some("v2")
            );

            let (status, body) = post_with_bearer(
                port,
                &format!(
                    "/api/v1/projects/project-a/prompts/support-reset/rollouts/{rollout_id}/apply"
                ),
                "",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout apply failed: {body}");
            let apply_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(apply_json["applied"].as_bool(), Some(true));
            assert_eq!(apply_json["rollout"]["status"].as_str(), Some("applied"));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v1_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v1_json["active"].as_bool(), Some(false));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v2",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v2_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v2_json["active"].as_bool(), Some(true));
        }

        #[tokio::test]
        async fn project_prompt_rollout_can_apply_canary_without_switching_active_version() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api.clone()).await;

            for (version, active) in [("v1", true), ("v2", false)] {
                api.upsert_project_prompt(ProjectPromptRecord {
                    project_id: "project-a".to_string(),
                    prompt_name: "support-reset".to_string(),
                    version: version.to_string(),
                    environment: "prod".to_string(),
                    description: Some(format!("Prompt {version}")),
                    target: "system".to_string(),
                    template_text: format!("Prompt version {version}"),
                    variables_schema_json: None,
                    rollout_metadata_json: None,
                    active,
                    updated_at: current_timestamp_string(),
                })
                .await
                .expect("governance enabled")
                .expect("upsert prompt");
            }
            api.upsert_project_prompt_rollout(ProjectPromptRolloutRecord {
                project_id: "project-a".to_string(),
                prompt_name: "support-reset".to_string(),
                rollout_id: "rollout-canary-1".to_string(),
                policy_name: "prod-strict".to_string(),
                baseline_version: Some("v1".to_string()),
                candidate_version: "v2".to_string(),
                baseline_run_id: "run-base".to_string(),
                candidate_run_id: "run-candidate".to_string(),
                target_environment: Some("prod".to_string()),
                status: "ready".to_string(),
                recommendation_action: Some("canary".to_string()),
                comparison_json: serde_json::json!({
                    "summary": {
                        "candidate_pass_rate": 1.0
                    }
                })
                .to_string(),
                created_at: current_timestamp_string(),
                applied_at: None,
            })
            .await
            .expect("governance enabled")
            .expect("upsert rollout");

            let apply_payload = serde_json::json!({
                "mode": "canary",
                "traffic_percent": 35
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-canary-1/apply",
                &apply_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt canary apply failed: {body}");
            let apply_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(apply_json["applied"].as_bool(), Some(true));
            assert_eq!(apply_json["mode"].as_str(), Some("canary"));
            assert_eq!(
                apply_json["rollout"]["status"].as_str(),
                Some("applied_canary")
            );
            assert_eq!(
                apply_json["rollout"]["runtime_rollout"]["mode"].as_str(),
                Some("canary")
            );
            assert_eq!(
                apply_json["rollout"]["runtime_rollout"]["traffic_percent"].as_u64(),
                Some(35)
            );

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v1_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v1_json["active"].as_bool(), Some(true));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v2",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v2_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v2_json["active"].as_bool(), Some(false));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-canary-1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout fetch failed: {body}");
            let rollout_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(rollout_json["status"].as_str(), Some("applied_canary"));
            assert_eq!(
                rollout_json["runtime_rollout"]["traffic_percent"].as_u64(),
                Some(35)
            );
        }

        #[tokio::test]
        async fn project_prompt_rollout_can_advance_through_policy_stages_and_auto_promote() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api.clone()).await;

            let rollout_policy_payload = serde_json::json!({
                "description": "Staged prod canary",
                "gate": {
                    "preset": "strict"
                },
                "canary": {
                    "steps": [10, 50, 100],
                    "auto_promote_final": true
                },
                "target_environment": "prod"
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/rollout-policies/prod-staged",
                &rollout_policy_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "rollout policy upsert failed: {body}");

            for (version, active) in [("v1", true), ("v2", false)] {
                api.upsert_project_prompt(ProjectPromptRecord {
                    project_id: "project-a".to_string(),
                    prompt_name: "support-reset".to_string(),
                    version: version.to_string(),
                    environment: "prod".to_string(),
                    description: Some(format!("Prompt {version}")),
                    target: "system".to_string(),
                    template_text: format!("Prompt version {version}"),
                    variables_schema_json: None,
                    rollout_metadata_json: None,
                    active,
                    updated_at: current_timestamp_string(),
                })
                .await
                .expect("governance enabled")
                .expect("upsert prompt");
            }
            api.upsert_project_prompt_rollout(ProjectPromptRolloutRecord {
                project_id: "project-a".to_string(),
                prompt_name: "support-reset".to_string(),
                rollout_id: "rollout-stage-1".to_string(),
                policy_name: "prod-staged".to_string(),
                baseline_version: Some("v1".to_string()),
                candidate_version: "v2".to_string(),
                baseline_run_id: "run-base".to_string(),
                candidate_run_id: "run-candidate".to_string(),
                target_environment: Some("prod".to_string()),
                status: "ready".to_string(),
                recommendation_action: Some("canary".to_string()),
                comparison_json: serde_json::json!({
                    "summary": {
                        "candidate_pass_rate": 1.0
                    }
                })
                .to_string(),
                created_at: current_timestamp_string(),
                applied_at: None,
            })
            .await
            .expect("governance enabled")
            .expect("upsert rollout");

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-stage-1/apply",
                &serde_json::json!({ "mode": "canary" }).to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt canary apply failed: {body}");
            let apply_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                apply_json["rollout"]["runtime_rollout"]["traffic_percent"].as_u64(),
                Some(10)
            );
            assert_eq!(
                apply_json["rollout"]["runtime_rollout"]["current_step_index"].as_u64(),
                Some(0)
            );

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-stage-1/advance",
                "",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout advance failed: {body}");
            let advance_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(advance_json["advanced"].as_bool(), Some(true));
            assert_eq!(advance_json["promoted"].as_bool(), Some(false));
            assert_eq!(
                advance_json["rollout"]["runtime_rollout"]["traffic_percent"].as_u64(),
                Some(50)
            );
            assert_eq!(
                advance_json["rollout"]["runtime_rollout"]["current_step_index"].as_u64(),
                Some(1)
            );

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-stage-1/advance",
                "",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout final advance failed: {body}");
            let final_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(final_json["advanced"].as_bool(), Some(true));
            assert_eq!(final_json["promoted"].as_bool(), Some(true));
            assert_eq!(final_json["mode"].as_str(), Some("promote"));
            assert_eq!(final_json["rollout"]["status"].as_str(), Some("applied"));
            assert!(final_json["rollout"]["runtime_rollout"].is_null());

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v1_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v1_json["active"].as_bool(), Some(false));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v2",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v2_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v2_json["active"].as_bool(), Some(true));
        }

        #[tokio::test]
        async fn project_prompt_rollout_can_rollback_live_canary() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api.clone()).await;

            let rollout_policy_payload = serde_json::json!({
                "description": "Rollbackable prod canary",
                "gate": {
                    "preset": "strict"
                },
                "canary": {
                    "steps": [20, 50]
                },
                "target_environment": "prod"
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/rollout-policies/prod-rollback",
                &rollout_policy_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "rollout policy upsert failed: {body}");

            for (version, active) in [("v1", true), ("v2", false)] {
                api.upsert_project_prompt(ProjectPromptRecord {
                    project_id: "project-a".to_string(),
                    prompt_name: "support-reset".to_string(),
                    version: version.to_string(),
                    environment: "prod".to_string(),
                    description: Some(format!("Prompt {version}")),
                    target: "system".to_string(),
                    template_text: format!("Prompt version {version}"),
                    variables_schema_json: None,
                    rollout_metadata_json: None,
                    active,
                    updated_at: current_timestamp_string(),
                })
                .await
                .expect("governance enabled")
                .expect("upsert prompt");
            }
            api.upsert_project_prompt_rollout(ProjectPromptRolloutRecord {
                project_id: "project-a".to_string(),
                prompt_name: "support-reset".to_string(),
                rollout_id: "rollout-stage-rollback".to_string(),
                policy_name: "prod-rollback".to_string(),
                baseline_version: Some("v1".to_string()),
                candidate_version: "v2".to_string(),
                baseline_run_id: "run-base".to_string(),
                candidate_run_id: "run-candidate".to_string(),
                target_environment: Some("prod".to_string()),
                status: "ready".to_string(),
                recommendation_action: Some("canary".to_string()),
                comparison_json: serde_json::json!({
                    "summary": {
                        "candidate_pass_rate": 1.0
                    }
                })
                .to_string(),
                created_at: current_timestamp_string(),
                applied_at: None,
            })
            .await
            .expect("governance enabled")
            .expect("upsert rollout");

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-stage-rollback/apply",
                &serde_json::json!({ "mode": "canary" }).to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt canary apply failed: {body}");

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-stage-rollback/rollback",
                "",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout rollback failed: {body}");
            let rollback_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(rollback_json["rolled_back"].as_bool(), Some(true));
            assert_eq!(
                rollback_json["rollout"]["status"].as_str(),
                Some("rolled_back")
            );
            assert!(rollback_json["rollout"]["runtime_rollout"].is_null());

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v1_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v1_json["active"].as_bool(), Some(true));

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/versions/v2",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let v2_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v2_json["active"].as_bool(), Some(false));
        }

        #[tokio::test]
        async fn project_prompt_rollout_can_auto_advance_live_canary_on_pass() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api.clone()).await;

            let rollout_policy_payload = serde_json::json!({
                "description": "Auto-advance prod canary",
                "gate": {
                    "preset": "strict"
                },
                "canary": {
                    "steps": [10, 50, 100],
                    "auto_advance_on_pass": true,
                    "auto_promote_final": true
                },
                "target_environment": "prod"
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/rollout-policies/prod-auto-advance",
                &rollout_policy_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "rollout policy upsert failed: {body}");

            seed_prompt_versions(&api).await;
            seed_ready_prompt_rollout(&api, "rollout-auto-advance", "prod-auto-advance").await;

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-auto-advance/apply",
                &serde_json::json!({ "mode": "canary" }).to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt canary apply failed: {body}");

            seed_rollout_eval_dataset(port, mgmt_token, "rollout-auto-advance").await;
            let baseline_port =
                start_static_eval_target_server("Use the reset password link from your email.")
                    .await;
            let candidate_port =
                start_static_eval_target_server("Use the reset password link from your email.")
                    .await;
            let baseline_run_id = execute_prompt_eval_run(
                port,
                mgmt_token,
                "rollout-auto-advance",
                baseline_port,
                "v1",
            )
            .await;
            let candidate_run_id = execute_prompt_eval_run(
                port,
                mgmt_token,
                "rollout-auto-advance",
                candidate_port,
                "v2",
            )
            .await;

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-auto-advance/evaluate",
                &serde_json::json!({
                    "baseline_run_id": baseline_run_id,
                    "candidate_run_id": candidate_run_id
                })
                .to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout evaluate failed: {body}");
            let evaluate_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(evaluate_json["evaluated"].as_bool(), Some(true));
            assert_eq!(evaluate_json["gate_passed"].as_bool(), Some(true));
            assert_eq!(evaluate_json["action"].as_str(), Some("advance"));
            assert_eq!(evaluate_json["applied"].as_bool(), Some(true));
            assert_eq!(
                evaluate_json["rollout"]["status"].as_str(),
                Some("applied_canary")
            );
            assert_eq!(
                evaluate_json["rollout"]["runtime_rollout"]["traffic_percent"].as_u64(),
                Some(50)
            );
            assert_eq!(
                evaluate_json["rollout"]["latest_canary_evaluation"]["action"].as_str(),
                Some("advance")
            );
            assert_eq!(
                evaluate_json["rollout"]["latest_canary_evaluation"]["comparison"]["gate"]
                    ["passed"]
                    .as_bool(),
                Some(true)
            );
        }

        #[tokio::test]
        async fn project_prompt_rollout_can_hold_live_canary_when_auto_advance_is_disabled() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api.clone()).await;

            let rollout_policy_payload = serde_json::json!({
                "description": "Manual-only prod canary",
                "gate": {
                    "preset": "strict"
                },
                "canary": {
                    "steps": [10, 50]
                },
                "target_environment": "prod"
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/rollout-policies/prod-manual-only",
                &rollout_policy_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "rollout policy upsert failed: {body}");

            seed_prompt_versions(&api).await;
            seed_ready_prompt_rollout(&api, "rollout-manual-only", "prod-manual-only").await;

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-manual-only/apply",
                &serde_json::json!({ "mode": "canary" }).to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt canary apply failed: {body}");

            seed_rollout_eval_dataset(port, mgmt_token, "rollout-manual-only").await;
            let baseline_port =
                start_static_eval_target_server("Use the reset password link from your email.")
                    .await;
            let candidate_port =
                start_static_eval_target_server("Use the reset password link from your email.")
                    .await;
            let baseline_run_id = execute_prompt_eval_run(
                port,
                mgmt_token,
                "rollout-manual-only",
                baseline_port,
                "v1",
            )
            .await;
            let candidate_run_id = execute_prompt_eval_run(
                port,
                mgmt_token,
                "rollout-manual-only",
                candidate_port,
                "v2",
            )
            .await;

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-manual-only/evaluate",
                &serde_json::json!({
                    "baseline_run_id": baseline_run_id,
                    "candidate_run_id": candidate_run_id
                })
                .to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout evaluate failed: {body}");
            let evaluate_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(evaluate_json["gate_passed"].as_bool(), Some(true));
            assert_eq!(evaluate_json["action"].as_str(), Some("hold"));
            assert_eq!(evaluate_json["applied"].as_bool(), Some(false));
            assert_eq!(
                evaluate_json["reason"].as_str(),
                Some("policy does not enable auto-advance on pass")
            );
            assert_eq!(
                evaluate_json["rollout"]["status"].as_str(),
                Some("applied_canary")
            );
            assert_eq!(
                evaluate_json["rollout"]["runtime_rollout"]["traffic_percent"].as_u64(),
                Some(10)
            );
            assert_eq!(
                evaluate_json["rollout"]["latest_canary_evaluation"]["action"].as_str(),
                Some("hold")
            );
        }

        #[tokio::test]
        async fn project_prompt_rollout_can_auto_rollback_live_canary_on_failed_gate() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api.clone()).await;

            let rollout_policy_payload = serde_json::json!({
                "description": "Auto-rollback prod canary",
                "gate": {
                    "preset": "strict"
                },
                "canary": {
                    "steps": [10, 50],
                    "auto_rollback_on_fail": true
                },
                "target_environment": "prod"
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/rollout-policies/prod-auto-rollback",
                &rollout_policy_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "rollout policy upsert failed: {body}");

            seed_prompt_versions(&api).await;
            seed_ready_prompt_rollout(&api, "rollout-auto-rollback", "prod-auto-rollback").await;

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-auto-rollback/apply",
                &serde_json::json!({ "mode": "canary" }).to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt canary apply failed: {body}");

            seed_rollout_eval_dataset(port, mgmt_token, "rollout-auto-rollback").await;
            let baseline_port =
                start_static_eval_target_server("Use the reset password link from your email.")
                    .await;
            let candidate_port =
                start_static_eval_target_server("Check your account settings page.").await;
            let baseline_run_id = execute_prompt_eval_run(
                port,
                mgmt_token,
                "rollout-auto-rollback",
                baseline_port,
                "v1",
            )
            .await;
            let candidate_run_id = execute_prompt_eval_run(
                port,
                mgmt_token,
                "rollout-auto-rollback",
                candidate_port,
                "v2",
            )
            .await;

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support-reset/rollouts/rollout-auto-rollback/evaluate",
                &serde_json::json!({
                    "baseline_run_id": baseline_run_id,
                    "candidate_run_id": candidate_run_id
                })
                .to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt rollout evaluate failed: {body}");
            let evaluate_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(evaluate_json["gate_passed"].as_bool(), Some(false));
            assert_eq!(evaluate_json["action"].as_str(), Some("rollback"));
            assert_eq!(evaluate_json["applied"].as_bool(), Some(true));
            assert_eq!(
                evaluate_json["rollout"]["status"].as_str(),
                Some("rolled_back")
            );
            assert!(evaluate_json["rollout"]["runtime_rollout"].is_null());
            assert_eq!(
                evaluate_json["rollout"]["latest_canary_evaluation"]["action"].as_str(),
                Some("rollback")
            );
            assert_eq!(
                evaluate_json["rollout"]["latest_canary_evaluation"]["comparison"]["gate"]
                    ["passed"]
                    .as_bool(),
                Some(false)
            );
        }

        #[tokio::test]
        async fn project_datasets_management_round_trip() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let dataset_payload = serde_json::json!({
                "description": "Replay set for support prompts",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "contains": "reset"
                },
                "metadata": {
                    "priority": "high"
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let dataset_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                dataset_json["dataset_name"].as_str(),
                Some("support-replay")
            );
            assert_eq!(
                dataset_json["description"].as_str(),
                Some("Replay set for support prompts")
            );

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay/items/case-1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let item_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(item_json["item_id"].as_str(), Some("case-1"));
            assert_eq!(
                item_json["input"]["messages"][0]["content"].as_str(),
                Some("Reset password")
            );
            assert_eq!(
                item_json["expected_output"]["contains"].as_str(),
                Some("reset")
            );

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay/items",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let items_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(items_json["items"].as_array().unwrap().len(), 1);

            let (status, body) =
                get_with_bearer(port, "/api/v1/projects/project-a/datasets", mgmt_token).await;
            assert_eq!(status, 200);
            assert!(body.contains("\"dataset_name\":\"support-replay\""));

            let (status, _) = delete_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay/items/case-1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);

            let (status, _) = delete_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);

            let (status, _) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 404);
        }

        #[tokio::test]
        async fn project_eval_runs_management_execute_and_persist_results() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let seen_requests = Arc::new(Mutex::new(Vec::<Value>::new()));
            let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let upstream_port = upstream_listener.local_addr().unwrap().port();
            let seen_requests_clone = Arc::clone(&seen_requests);
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match upstream_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    let seen_requests = Arc::clone(&seen_requests_clone);
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| {
                            let seen_requests = Arc::clone(&seen_requests);
                            async move {
                                let body = req.into_body().collect().await.unwrap().to_bytes();
                                let request_json: Value = serde_json::from_slice(&body).unwrap();
                                seen_requests.lock().unwrap().push(request_json);
                                let response = serde_json::json!({
                                    "choices": [
                                        {
                                            "message": {
                                                "role": "assistant",
                                                "content": "Reset your password using the emailed reset link."
                                            }
                                        }
                                    ],
                                    "usage": {
                                        "prompt_tokens": 11,
                                        "completion_tokens": 5,
                                        "total_cost": 0.012
                                    }
                                });
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(200)
                                        .header("content-type", "application/json")
                                        .body(Full::new(Bytes::from(response.to_string())))
                                        .unwrap(),
                                )
                            }
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let dataset_payload = serde_json::json!({
                "description": "Replay set for support prompts",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "contains": "reset"
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            let eval_payload = serde_json::json!({
                "dataset_name": "support-replay",
                "target_url": format!("http://127.0.0.1:{upstream_port}/v1/chat/completions"),
                "timeout_ms": 1000
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "eval run execution failed: {body}");
            let eval_json: Value = serde_json::from_str(&body).unwrap();
            let run_id = eval_json["run"]["run_id"]
                .as_str()
                .expect("run id present")
                .to_string();
            assert_eq!(eval_json["run"]["status"].as_str(), Some("completed"));
            assert_eq!(eval_json["run"]["passed_items"].as_u64(), Some(1));
            assert_eq!(eval_json["run"]["failed_items"].as_u64(), Some(0));
            assert_eq!(eval_json["items"].as_array().unwrap().len(), 1);
            assert_eq!(
                eval_json["items"][0]["output_text"].as_str(),
                Some("Reset your password using the emailed reset link.")
            );
            assert_eq!(eval_json["items"][0]["input_tokens"].as_u64(), Some(11));
            assert_eq!(eval_json["items"][0]["output_tokens"].as_u64(), Some(5));

            let (status, body) =
                get_with_bearer(port, "/api/v1/projects/project-a/eval-runs", mgmt_token).await;
            assert_eq!(status, 200);
            let runs_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(runs_json["runs"].as_array().unwrap().len(), 1);
            assert_eq!(
                runs_json["runs"][0]["run_id"].as_str(),
                Some(run_id.as_str())
            );

            let (status, body) = get_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/eval-runs/{run_id}/items"),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let items_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(items_json["items"].as_array().unwrap().len(), 1);
            assert_eq!(items_json["items"][0]["passed"].as_bool(), Some(true));

            let seen = seen_requests.lock().unwrap();
            assert_eq!(seen.len(), 1);
            assert_eq!(
                seen[0]["messages"][0]["content"].as_str(),
                Some("Reset password")
            );
        }

        #[tokio::test]
        async fn project_eval_runs_support_async_execution_and_comparison() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let seen_requests = Arc::new(Mutex::new(Vec::<Value>::new()));
            let request_index = Arc::new(AtomicUsize::new(0));
            let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let upstream_port = upstream_listener.local_addr().unwrap().port();
            let seen_requests_clone = Arc::clone(&seen_requests);
            let request_index_clone = Arc::clone(&request_index);
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match upstream_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    let seen_requests = Arc::clone(&seen_requests_clone);
                    let request_index = Arc::clone(&request_index_clone);
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| {
                            let seen_requests = Arc::clone(&seen_requests);
                            let request_index = Arc::clone(&request_index);
                            async move {
                                let body = req.into_body().collect().await.unwrap().to_bytes();
                                let request_json: Value = serde_json::from_slice(&body).unwrap();
                                seen_requests.lock().unwrap().push(request_json);
                                let index = request_index.fetch_add(1, Ordering::Relaxed);
                                tokio::time::sleep(Duration::from_millis(50)).await;
                                let content = if index == 0 {
                                    "Use reset link 1234."
                                } else {
                                    "Contact support."
                                };
                                let response = serde_json::json!({
                                    "choices": [
                                        {
                                            "message": {
                                                "role": "assistant",
                                                "content": content
                                            }
                                        }
                                    ],
                                    "usage": {
                                        "prompt_tokens": 9,
                                        "completion_tokens": 4,
                                        "total_cost": 0.008
                                    }
                                });
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(200)
                                        .header("content-type", "application/json")
                                        .body(Full::new(Bytes::from(response.to_string())))
                                        .unwrap(),
                                )
                            }
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let dataset_payload = serde_json::json!({
                "description": "Replay set for eval v2",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay-v2",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "contains": "reset",
                    "not_contains": "support",
                    "starts_with": "use",
                    "ends_with": "1234.",
                    "regex": "reset link \\d{4}\\.$",
                    "status_code": 200
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/support-replay-v2/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            let async_eval_payload = serde_json::json!({
                "dataset_name": "support-replay-v2",
                "target_url": format!("http://127.0.0.1:{upstream_port}/v1/chat/completions"),
                "timeout_ms": 1000,
                "prompt_name": "support-reset",
                "prompt_version": "prod-1",
                "provider_name": "openai",
                "model": "gpt-4o-mini",
                "route_path": "/v1/chat/completions",
                "safety_profile": "standard",
                "async": true
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &async_eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 202, "async eval run queue failed: {body}");
            let queued_json: Value = serde_json::from_str(&body).unwrap();
            let baseline_run_id = queued_json["run"]["run_id"]
                .as_str()
                .expect("baseline run id")
                .to_string();
            assert_eq!(queued_json["run"]["status"].as_str(), Some("queued"));
            assert_eq!(queued_json["queued"].as_bool(), Some(true));

            let mut completed_run = None;
            for _ in 0..40 {
                let (status, body) = get_with_bearer(
                    port,
                    &format!("/api/v1/projects/project-a/eval-runs/{baseline_run_id}"),
                    mgmt_token,
                )
                .await;
                assert_eq!(status, 200);
                let run_json: Value = serde_json::from_str(&body).unwrap();
                if run_json["status"].as_str() == Some("completed") {
                    completed_run = Some(run_json);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            let completed_run = completed_run.expect("async eval run completed");
            assert_eq!(completed_run["passed_items"].as_u64(), Some(1));
            assert_eq!(
                completed_run["summary"]["context"]["prompt_name"].as_str(),
                Some("support-reset")
            );
            assert_eq!(
                completed_run["summary"]["context"]["provider_name"].as_str(),
                Some("openai")
            );

            let (status, body) = get_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/eval-runs/{baseline_run_id}/items"),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let items_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(items_json["items"][0]["passed"].as_bool(), Some(true));
            assert_eq!(
                items_json["items"][0]["evaluation"]["regex"].as_str(),
                Some("reset link \\d{4}\\.$")
            );

            let sync_eval_payload = serde_json::json!({
                "dataset_name": "support-replay-v2",
                "target_url": format!("http://127.0.0.1:{upstream_port}/v1/chat/completions"),
                "timeout_ms": 1000,
                "prompt_name": "support-reset",
                "prompt_version": "candidate-2",
                "provider_name": "anthropic",
                "model": "claude-3-5-sonnet",
                "route_path": "/v1/messages",
                "safety_profile": "strict"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &sync_eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "sync eval run failed: {body}");
            let candidate_json: Value = serde_json::from_str(&body).unwrap();
            let candidate_run_id = candidate_json["run"]["run_id"]
                .as_str()
                .expect("candidate run id")
                .to_string();
            assert_eq!(candidate_json["run"]["passed_items"].as_u64(), Some(0));

            let (status, body) = get_with_bearer(
                port,
                &format!(
                    "/api/v1/projects/project-a/eval-runs/compare?baseline_run_id={baseline_run_id}&candidate_run_id={candidate_run_id}"
                ),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "comparison failed: {body}");
            let comparison_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                comparison_json["summary"]["regressed_items"].as_u64(),
                Some(1)
            );
            assert_eq!(
                comparison_json["summary"]["improved_items"].as_u64(),
                Some(0)
            );
            assert_eq!(comparison_json["items"].as_array().unwrap().len(), 1);
            assert_eq!(
                comparison_json["items"][0]["baseline_passed"].as_bool(),
                Some(true)
            );
            assert_eq!(
                comparison_json["items"][0]["candidate_passed"].as_bool(),
                Some(false)
            );
            assert_eq!(
                comparison_json["items"][0]["regressed"].as_bool(),
                Some(true)
            );
            assert_eq!(
                comparison_json["context"]["baseline"]["prompt_version"].as_str(),
                Some("prod-1")
            );
            assert_eq!(
                comparison_json["context"]["candidate"]["prompt_version"].as_str(),
                Some("candidate-2")
            );
            assert!(comparison_json["context"]["changed_fields"]
                .as_array()
                .unwrap()
                .iter()
                .any(|field| field.as_str() == Some("prompt_version")));
            assert!(comparison_json["context"]["changed_fields"]
                .as_array()
                .unwrap()
                .iter()
                .any(|field| field.as_str() == Some("provider_name")));
            assert!(comparison_json["gate"].is_null());

            let (status, body) = get_with_bearer(
                port,
                &format!(
                    "/api/v1/projects/project-a/eval-runs/compare?baseline_run_id={baseline_run_id}&candidate_run_id={candidate_run_id}&max_regressions=0&min_candidate_pass_rate=0.9"
                ),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "gated comparison failed: {body}");
            let gated_comparison_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                gated_comparison_json["gate"]["passed"].as_bool(),
                Some(false)
            );
            assert_eq!(
                gated_comparison_json["gate"]["thresholds"]["max_regressions"].as_u64(),
                Some(0)
            );
            assert!(
                gated_comparison_json["gate"]["reasons"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|reason| reason
                        .as_str()
                        .unwrap_or_default()
                        .contains("regressed_items")),
                "unexpected gate body: {body}"
            );

            let (status, body) = get_with_bearer(
                port,
                &format!(
                    "/api/v1/projects/project-a/eval-runs/compare?baseline_run_id={baseline_run_id}&candidate_run_id={candidate_run_id}&preset=strict"
                ),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "preset comparison failed: {body}");
            let preset_comparison_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                preset_comparison_json["gate"]["preset"].as_str(),
                Some("strict")
            );
            assert_eq!(
                preset_comparison_json["gate"]["thresholds"]["max_regressions"].as_u64(),
                Some(0)
            );
            assert_eq!(
                preset_comparison_json["gate"]["thresholds"]["max_latency_increase_ms"].as_f64(),
                Some(25.0)
            );
            assert_eq!(
                preset_comparison_json["gate"]["recommendation"]["action"].as_str(),
                Some("hold")
            );
            assert!(preset_comparison_json["gate"]["recommendation"]["summary"]
                .as_str()
                .unwrap_or_default()
                .contains("strict rollout gate"));
            assert!(
                preset_comparison_json["gate"]["recommendation"]["changed_context_fields"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|field| field.as_str() == Some("route_path"))
            );

            let seen = seen_requests.lock().unwrap();
            assert_eq!(seen.len(), 2);
            assert_eq!(
                seen[0]["messages"][0]["content"].as_str(),
                Some("Reset password")
            );
        }

        #[tokio::test]
        async fn project_eval_runs_support_structured_output_matchers() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let upstream_port = upstream_listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match upstream_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| async move {
                            let _body = req.into_body().collect().await.unwrap().to_bytes();
                            let response = serde_json::json!({
                                "output_json": {
                                    "decision": {
                                        "approved": true,
                                        "reason": "Reset link sent to the account owner"
                                    },
                                    "citations": ["kb-123"]
                                },
                                "usage": {
                                    "prompt_tokens": 12,
                                    "completion_tokens": 5,
                                    "total_cost": 0.003
                                }
                            });
                            Ok::<_, hyper::Error>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(response.to_string())))
                                    .unwrap(),
                            )
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let dataset_payload = serde_json::json!({
                "description": "Structured eval dataset",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/structured-replay",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "json_path_exists": ["decision.approved", "decision.reason", "citations.0"],
                    "json_path_equals": {
                        "decision.approved": true,
                        "citations.0": "kb-123"
                    },
                    "json_path_contains": {
                        "decision.reason": "reset link"
                    },
                    "status_code": 200
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/structured-replay/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            let eval_payload = serde_json::json!({
                "dataset_name": "structured-replay",
                "target_url": format!("http://127.0.0.1:{upstream_port}/v1/responses"),
                "timeout_ms": 1000,
                "provider_name": "openai",
                "model": "gpt-4o-mini"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "structured eval run failed: {body}");
            let run_json: Value = serde_json::from_str(&body).unwrap();
            let run_id = run_json["run"]["run_id"]
                .as_str()
                .expect("run id present")
                .to_string();
            assert_eq!(run_json["run"]["passed_items"].as_u64(), Some(1));

            let (status, body) = get_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/eval-runs/{run_id}/items"),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "structured eval items fetch failed: {body}");
            let items_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(items_json["items"][0]["passed"].as_bool(), Some(true));
            assert_eq!(
                items_json["items"][0]["evaluation"]["structured_output_source"].as_str(),
                Some("output_json")
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["json_path_equals_results"]
                    ["decision.approved"]["passed"]
                    .as_bool(),
                Some(true)
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["json_path_contains_results"]
                    ["decision.reason"]["passed"]
                    .as_bool(),
                Some(true)
            );
        }

        #[tokio::test]
        async fn startup_recovery_replays_running_eval_run_without_double_counting() {
            let temp = NamedTempFile::new().unwrap();
            let store_url = format!("sqlite://{}", temp.path().display());
            let store = store::connect(&store_url).await.unwrap();

            store
                .upsert_project_dataset(&ProjectDatasetRecord {
                    project_id: "project-a".to_string(),
                    dataset_name: "recover-replay".to_string(),
                    description: Some("Recovery eval dataset".to_string()),
                    schema_json: None,
                    updated_at: "1".to_string(),
                })
                .await
                .unwrap();
            store
                .upsert_project_dataset_item(&ProjectDatasetItemRecord {
                    project_id: "project-a".to_string(),
                    dataset_name: "recover-replay".to_string(),
                    item_id: "case-1".to_string(),
                    input_json: serde_json::json!({
                        "messages": [{"role":"user","content":"Reset password"}]
                    })
                    .to_string(),
                    expected_output_json: Some(
                        serde_json::json!({"contains":"reset link"}).to_string(),
                    ),
                    metadata_json: None,
                    updated_at: "1".to_string(),
                })
                .await
                .unwrap();

            let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let upstream_port = upstream_listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match upstream_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| async move {
                            let _body = req.into_body().collect().await.unwrap().to_bytes();
                            let response = serde_json::json!({
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": "Use reset link 4321."
                                        }
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 7,
                                    "completion_tokens": 4,
                                    "total_cost": 0.004
                                }
                            });
                            Ok::<_, hyper::Error>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(response.to_string())))
                                    .unwrap(),
                            )
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            store
                .upsert_project_eval_run(&ProjectEvalRunRecord {
                    run_id: "eval-recover-1".to_string(),
                    project_id: "project-a".to_string(),
                    dataset_name: "recover-replay".to_string(),
                    target_url: format!("http://127.0.0.1:{upstream_port}/v1/chat/completions"),
                    status: "running".to_string(),
                    total_items: 1,
                    passed_items: 1,
                    failed_items: 0,
                    total_input_tokens: 999,
                    total_output_tokens: 888,
                    total_cost: 9.99,
                    average_latency_ms: 777.0,
                    summary_json: Some(
                        serde_json::json!({
                            "pass_rate": 1.0,
                            "timeout_ms": 1000,
                            "context": {
                                "provider_name": "openai",
                                "model": "gpt-4o-mini"
                            },
                            "request": {
                                "headers": {}
                            }
                        })
                        .to_string(),
                    ),
                    created_at: "1".to_string(),
                    completed_at: None,
                })
                .await
                .unwrap();
            store
                .upsert_project_eval_run_item(&ProjectEvalRunItemRecord {
                    run_id: "eval-recover-1".to_string(),
                    project_id: "project-a".to_string(),
                    dataset_name: "recover-replay".to_string(),
                    item_id: "case-1".to_string(),
                    passed: false,
                    status_code: Some(500),
                    latency_ms: 1,
                    output_text: Some("stale output".to_string()),
                    evaluation_json: Some(
                        serde_json::json!({"kind":"stale","passed":false}).to_string(),
                    ),
                    error: Some("stale".to_string()),
                    input_tokens: 1,
                    output_tokens: 1,
                    cost: 1.0,
                    created_at: "1".to_string(),
                })
                .await
                .unwrap();
            drop(store);

            let (_plugins, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some(&store_url),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some("recover-admin".to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();

            let mut recovered_run = None;
            for _ in 0..80 {
                let run = api
                    .get_project_eval_run("project-a", "eval-recover-1")
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                if run.status == "completed" {
                    recovered_run = Some(run);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }

            let recovered_run = recovered_run.expect("recovered eval run completed");
            assert_eq!(recovered_run.passed_items, 1);
            assert_eq!(recovered_run.failed_items, 0);
            assert_eq!(recovered_run.total_input_tokens, 7);
            assert_eq!(recovered_run.total_output_tokens, 4);
            assert!((recovered_run.total_cost - 0.004).abs() < 1e-9);
            assert!(
                recovered_run.average_latency_ms < 777.0,
                "stale latency leaked into recovered run: {:?}",
                recovered_run
            );

            let items = api
                .list_project_eval_run_items("project-a", "eval-recover-1")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(items.len(), 1);
            assert!(items[0].passed);
            assert_eq!(
                items[0].output_text.as_deref(),
                Some("Use reset link 4321.")
            );
        }

        #[tokio::test]
        async fn project_eval_runs_support_external_judge_evaluator() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let judge_requests = Arc::new(Mutex::new(Vec::<Value>::new()));

            let target_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let target_port = target_listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match target_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| async move {
                            let _body = req.into_body().collect().await.unwrap().to_bytes();
                            let response = serde_json::json!({
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": "Use the reset password link from the email we just sent."
                                        }
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 8,
                                    "completion_tokens": 7,
                                    "total_cost": 0.006
                                }
                            });
                            Ok::<_, hyper::Error>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(response.to_string())))
                                    .unwrap(),
                            )
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let judge_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let judge_port = judge_listener.local_addr().unwrap().port();
            let judge_requests_clone = Arc::clone(&judge_requests);
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match judge_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    let judge_requests = Arc::clone(&judge_requests_clone);
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| {
                            let judge_requests = Arc::clone(&judge_requests);
                            async move {
                                let body = req.into_body().collect().await.unwrap().to_bytes();
                                let request_json: Value = serde_json::from_slice(&body).unwrap();
                                judge_requests.lock().unwrap().push(request_json.clone());
                                let output_text =
                                    request_json["output_text"].as_str().unwrap_or("");
                                let passed = output_text.contains("reset password");
                                let response = serde_json::json!({
                                    "passed": passed,
                                    "score": if passed { 0.95 } else { 0.1 },
                                    "reasoning": if passed {
                                        "The answer directly instructs the user to use the reset password link."
                                    } else {
                                        "The answer does not solve the reset-password task."
                                    }
                                });
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(200)
                                        .header("content-type", "application/json")
                                        .body(Full::new(Bytes::from(response.to_string())))
                                        .unwrap(),
                                )
                            }
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let dataset_payload = serde_json::json!({
                "description": "Judge-based eval dataset",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/judge-replay",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "judge": {
                        "rubric": "Does the answer tell the user how to reset the password?",
                        "min_score": 0.8
                    },
                    "status_code": 200
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/judge-replay/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            let eval_payload = serde_json::json!({
                "dataset_name": "judge-replay",
                "target_url": format!("http://127.0.0.1:{target_port}/v1/chat/completions"),
                "judge_url": format!("http://127.0.0.1:{judge_port}/judge"),
                "judge_timeout_ms": 1000,
                "timeout_ms": 1000,
                "provider_name": "openai",
                "model": "gpt-4o-mini"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "judge eval run failed: {body}");
            let run_json: Value = serde_json::from_str(&body).unwrap();
            let run_id = run_json["run"]["run_id"]
                .as_str()
                .expect("run id present")
                .to_string();
            assert_eq!(run_json["run"]["passed_items"].as_u64(), Some(1));

            let (status, body) = get_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/eval-runs/{run_id}/items"),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "judge eval items fetch failed: {body}");
            let items_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(items_json["items"][0]["passed"].as_bool(), Some(true));
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["passed"].as_bool(),
                Some(true)
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["response"]["score"].as_f64(),
                Some(0.95)
            );

            let judge_requests = judge_requests.lock().unwrap();
            assert_eq!(judge_requests.len(), 1);
            assert_eq!(
                judge_requests[0]["judge"]["rubric"].as_str(),
                Some("Does the answer tell the user how to reset the password?")
            );
            assert_eq!(
                judge_requests[0]["context"]["provider_name"].as_str(),
                Some("openai")
            );
            assert!(judge_requests[0]["output_text"]
                .as_str()
                .unwrap_or_default()
                .contains("reset password"));
        }

        #[tokio::test]
        async fn project_eval_runs_support_openai_judge_evaluator() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let judge_requests = Arc::new(Mutex::new(Vec::<Value>::new()));

            let target_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let target_port = target_listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match target_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| async move {
                            let _body = req.into_body().collect().await.unwrap().to_bytes();
                            let response = serde_json::json!({
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": "Use the reset password link from the email we just sent."
                                        }
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 8,
                                    "completion_tokens": 7,
                                    "total_cost": 0.006
                                }
                            });
                            Ok::<_, hyper::Error>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(response.to_string())))
                                    .unwrap(),
                            )
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let judge_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let judge_port = judge_listener.local_addr().unwrap().port();
            let judge_requests_clone = Arc::clone(&judge_requests);
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match judge_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    let judge_requests = Arc::clone(&judge_requests_clone);
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| {
                            let judge_requests = Arc::clone(&judge_requests);
                            async move {
                                assert_eq!(
                                    req.headers()
                                        .get("authorization")
                                        .and_then(|value| value.to_str().ok()),
                                    Some("Bearer judge-secret")
                                );
                                let body = req.into_body().collect().await.unwrap().to_bytes();
                                let request_json: Value = serde_json::from_slice(&body).unwrap();
                                judge_requests.lock().unwrap().push(request_json.clone());
                                let response = serde_json::json!({
                                    "id": "chatcmpl-judge-1",
                                    "object": "chat.completion",
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "{\"passed\":true,\"score\":0.92,\"reasoning\":\"The answer tells the user to use the reset password link.\"}"
                                        },
                                        "finish_reason": "stop"
                                    }]
                                });
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(200)
                                        .header("content-type", "application/json")
                                        .body(Full::new(Bytes::from(response.to_string())))
                                        .unwrap(),
                                )
                            }
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let dataset_payload = serde_json::json!({
                "description": "OpenAI judge eval dataset",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/judge-openai",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "judge": {
                        "rubric": "Does the answer tell the user how to reset the password?",
                        "min_score": 0.8
                    },
                    "status_code": 200
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/judge-openai/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            let eval_payload = serde_json::json!({
                "dataset_name": "judge-openai",
                "target_url": format!("http://127.0.0.1:{target_port}/v1/chat/completions"),
                "judge_url": format!("http://127.0.0.1:{judge_port}/v1/chat/completions"),
                "judge_kind": "openai",
                "judge_model": "gpt-4o-mini",
                "judge_headers": {
                    "authorization": "Bearer judge-secret"
                },
                "judge_timeout_ms": 1000,
                "timeout_ms": 1000,
                "provider_name": "openai",
                "model": "gpt-4o-mini"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "OpenAI judge eval run failed: {body}");
            let run_json: Value = serde_json::from_str(&body).unwrap();
            let run_id = run_json["run"]["run_id"]
                .as_str()
                .expect("run id present")
                .to_string();
            assert_eq!(run_json["run"]["passed_items"].as_u64(), Some(1));

            let (status, body) = get_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/eval-runs/{run_id}/items"),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "OpenAI judge eval items fetch failed: {body}");
            let items_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(items_json["items"][0]["passed"].as_bool(), Some(true));
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["kind"].as_str(),
                Some("judge_openai")
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["model"].as_str(),
                Some("gpt-4o-mini")
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["passed"].as_bool(),
                Some(true)
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["parsed_response"]["score"].as_f64(),
                Some(0.92)
            );

            let judge_requests = judge_requests.lock().unwrap();
            assert_eq!(judge_requests.len(), 1);
            assert_eq!(judge_requests[0]["model"].as_str(), Some("gpt-4o-mini"));
            assert_eq!(
                judge_requests[0]["messages"][0]["role"].as_str(),
                Some("system")
            );
            assert!(judge_requests[0]["messages"][1]["content"]
                .as_str()
                .unwrap_or_default()
                .contains("reset password"));
        }

        #[tokio::test]
        async fn project_eval_runs_support_anthropic_judge_evaluator() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let judge_requests = Arc::new(Mutex::new(Vec::<Value>::new()));

            let target_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let target_port = target_listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match target_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| async move {
                            let _body = req.into_body().collect().await.unwrap().to_bytes();
                            let response = serde_json::json!({
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": "Use the reset password link from the email we just sent."
                                        }
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 8,
                                    "completion_tokens": 7,
                                    "total_cost": 0.006
                                }
                            });
                            Ok::<_, hyper::Error>(
                                Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(response.to_string())))
                                    .unwrap(),
                            )
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let judge_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let judge_port = judge_listener.local_addr().unwrap().port();
            let judge_requests_clone = Arc::clone(&judge_requests);
            tokio::spawn(async move {
                use hyper::body::Incoming;
                use hyper::service::service_fn;
                use hyper::{Request, Response};
                use hyper_util::rt::TokioIo;

                loop {
                    let (stream, _) = match judge_listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => continue,
                    };
                    let judge_requests = Arc::clone(&judge_requests_clone);
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = service_fn(move |req: Request<Incoming>| {
                            let judge_requests = Arc::clone(&judge_requests);
                            async move {
                                assert_eq!(
                                    req.headers()
                                        .get("x-api-key")
                                        .and_then(|value| value.to_str().ok()),
                                    Some("judge-secret")
                                );
                                assert_eq!(
                                    req.headers()
                                        .get("anthropic-version")
                                        .and_then(|value| value.to_str().ok()),
                                    Some("2023-06-01")
                                );
                                let body = req.into_body().collect().await.unwrap().to_bytes();
                                let request_json: Value = serde_json::from_slice(&body).unwrap();
                                judge_requests.lock().unwrap().push(request_json.clone());
                                let response = serde_json::json!({
                                    "id": "msg-judge-1",
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{
                                        "type": "text",
                                        "text": "{\"passed\":true,\"score\":0.91,\"reasoning\":\"The answer tells the user to use the reset password link.\"}"
                                    }]
                                });
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(200)
                                        .header("content-type", "application/json")
                                        .body(Full::new(Bytes::from(response.to_string())))
                                        .unwrap(),
                                )
                            }
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc)
                            .await;
                    });
                }
            });

            let dataset_payload = serde_json::json!({
                "description": "Anthropic judge eval dataset",
                "schema": {
                    "type": "object",
                    "properties": {
                        "messages": { "type": "array" }
                    }
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/judge-anthropic",
                &dataset_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset upsert failed: {body}");

            let item_payload = serde_json::json!({
                "input": {
                    "messages": [{"role":"user","content":"Reset password"}]
                },
                "expected_output": {
                    "judge": {
                        "rubric": "Does the answer tell the user how to reset the password?",
                        "min_score": 0.8
                    },
                    "status_code": 200
                }
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/datasets/judge-anthropic/items/case-1",
                &item_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "dataset item upsert failed: {body}");

            let eval_payload = serde_json::json!({
                "dataset_name": "judge-anthropic",
                "target_url": format!("http://127.0.0.1:{target_port}/v1/chat/completions"),
                "judge_url": format!("http://127.0.0.1:{judge_port}/v1/messages"),
                "judge_kind": "anthropic",
                "judge_model": "claude-3-5-sonnet-20241022",
                "judge_headers": {
                    "x-api-key": "judge-secret",
                    "anthropic-version": "2023-06-01"
                },
                "judge_timeout_ms": 1000,
                "timeout_ms": 1000,
                "provider_name": "openai",
                "model": "gpt-4o-mini"
            })
            .to_string();
            let (status, body) = post_with_bearer(
                port,
                "/api/v1/projects/project-a/eval-runs",
                &eval_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "Anthropic judge eval run failed: {body}");
            let run_json: Value = serde_json::from_str(&body).unwrap();
            let run_id = run_json["run"]["run_id"]
                .as_str()
                .expect("run id present")
                .to_string();
            assert_eq!(run_json["run"]["passed_items"].as_u64(), Some(1));

            let (status, body) = get_with_bearer(
                port,
                &format!("/api/v1/projects/project-a/eval-runs/{run_id}/items"),
                mgmt_token,
            )
            .await;
            assert_eq!(
                status, 200,
                "Anthropic judge eval items fetch failed: {body}"
            );
            let items_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(items_json["items"][0]["passed"].as_bool(), Some(true));
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["kind"].as_str(),
                Some("judge_anthropic")
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["model"].as_str(),
                Some("claude-3-5-sonnet-20241022")
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["passed"].as_bool(),
                Some(true)
            );
            assert_eq!(
                items_json["items"][0]["evaluation"]["judge"]["parsed_response"]["score"].as_f64(),
                Some(0.91)
            );

            let judge_requests = judge_requests.lock().unwrap();
            assert_eq!(judge_requests.len(), 1);
            assert_eq!(
                judge_requests[0]["model"].as_str(),
                Some("claude-3-5-sonnet-20241022")
            );
            assert_eq!(
                judge_requests[0]["messages"][0]["role"].as_str(),
                Some("user")
            );
            assert!(judge_requests[0]["system"]
                .as_str()
                .unwrap_or_default()
                .contains("Return JSON only"));
            assert!(judge_requests[0]["messages"][0]["content"]
                .as_str()
                .unwrap_or_default()
                .contains("reset password"));
        }

        #[tokio::test]
        async fn project_governance_history_management_round_trip() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let tool_v1 = serde_json::json!({
                "description": "Search docs v1",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                },
                "executor_kind": "webhook",
                "executor_config": {
                    "url": "http://tool.local/search",
                    "method": "POST"
                },
                "enabled": true
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/tools/web_search",
                &tool_v1,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "tool v1 upsert failed: {body}");

            let tool_v2 = serde_json::json!({
                "description": "Search docs v2",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                },
                "executor_kind": "webhook",
                "executor_config": {
                    "url": "http://tool.local/search-v2",
                    "method": "POST"
                },
                "enabled": true
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/tools/web_search",
                &tool_v2,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "tool v2 upsert failed: {body}");

            let prompt_payload = serde_json::json!({
                "environment": "prod",
                "target": "system",
                "template_text": "You are helping {{customer}}.",
                "active": true
            })
            .to_string();
            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/prompts/support/versions/v1",
                &prompt_payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "prompt upsert failed: {body}");

            let (status, body) = delete_with_bearer(
                port,
                "/api/v1/projects/project-a/tools/web_search",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "tool delete failed: {body}");

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/history?include_diff=1",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let history_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(history_json["project_id"].as_str(), Some("project-a"));
            let changes = history_json["changes"].as_array().expect("changes array");
            assert!(changes.len() >= 4, "unexpected history body: {body}");

            let tool_delete = changes
                .iter()
                .find(|change| {
                    change["resource_type"].as_str() == Some("project_tool")
                        && change["resource_id"].as_str() == Some("web_search")
                        && change["action"].as_str() == Some("delete")
                })
                .expect("project tool delete history");
            assert_eq!(tool_delete["after"], Value::Null);
            assert_eq!(
                tool_delete["before"]["tool_name"].as_str(),
                Some("web_search")
            );
            assert_eq!(
                tool_delete["before"]["description"].as_str(),
                Some("Search docs v2")
            );

            let tool_upsert = changes
                .iter()
                .find(|change| {
                    change["resource_type"].as_str() == Some("project_tool")
                        && change["resource_id"].as_str() == Some("web_search")
                        && change["action"].as_str() == Some("upsert")
                        && change["after"]["description"].as_str() == Some("Search docs v2")
                })
                .expect("project tool update history");
            assert_eq!(
                tool_upsert["before"]["description"].as_str(),
                Some("Search docs v1")
            );
            assert_eq!(
                tool_upsert["diff"]["description"]["before"].as_str(),
                Some("Search docs v1")
            );
            assert_eq!(
                tool_upsert["diff"]["description"]["after"].as_str(),
                Some("Search docs v2")
            );
            assert_eq!(
                tool_upsert["diff"]["executor_config"]["url"]["before"].as_str(),
                Some("http://tool.local/search")
            );
            assert_eq!(
                tool_upsert["diff"]["executor_config"]["url"]["after"].as_str(),
                Some("http://tool.local/search-v2")
            );

            let prompt_upsert = changes
                .iter()
                .find(|change| {
                    change["resource_type"].as_str() == Some("project_prompt")
                        && change["resource_id"].as_str() == Some("support:v1")
                        && change["action"].as_str() == Some("upsert")
                })
                .expect("project prompt history");
            assert_eq!(prompt_upsert["before"], Value::Null);
            assert_eq!(
                prompt_upsert["after"]["prompt_name"].as_str(),
                Some("support")
            );
            assert_eq!(prompt_upsert["after"]["version"].as_str(), Some("v1"));
            assert_eq!(prompt_upsert["diff"]["before"], Value::Null);
            assert_eq!(
                prompt_upsert["diff"]["after"]["prompt_name"].as_str(),
                Some("support")
            );

            let (status, body) = get_with_bearer(
                port,
                "/api/v1/projects/project-a/history?resource_type=project_tool&limit=2",
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200);
            let filtered_json: Value = serde_json::from_str(&body).unwrap();
            let filtered = filtered_json["changes"]
                .as_array()
                .expect("filtered changes");
            assert_eq!(filtered.len(), 2);
            assert!(filtered
                .iter()
                .all(|change| change["resource_type"].as_str() == Some("project_tool")));
        }

        #[tokio::test]
        async fn project_policy_management_round_trip_includes_timeout_settings() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let payload = serde_json::json!({
                "adaptive_enabled": true,
                "timeout_secs": 45,
                "provider_rpm_limits": {
                    "openai": 88,
                    "anthropic": 66
                },
                "provider_tpm_limits": {
                    "openai": 8800,
                    "anthropic": 6600
                },
                "provider_timeouts": {
                    "openai": 15,
                    "anthropic": 12
                },
                "provider_input_costs": {
                    "openai": 0.03,
                    "anthropic": 0.01
                },
                "semantic_cache_enabled": true,
                "semantic_cache_ttl_secs": 900,
                "semantic_cache_similarity_threshold": 0.72
            })
            .to_string();

            let (status, body) = put_with_bearer(
                port,
                "/api/v1/projects/project-a/policy",
                &payload,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "policy upsert failed: {body}");

            let (status, body) =
                get_with_bearer(port, "/api/v1/projects/project-a/policy", mgmt_token).await;
            assert_eq!(status, 200);
            let policy_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(policy_json["timeout_secs"].as_u64(), Some(45));
            assert_eq!(
                policy_json["provider_rpm_limits"]["openai"].as_u64(),
                Some(88)
            );
            assert_eq!(
                policy_json["provider_rpm_limits"]["anthropic"].as_u64(),
                Some(66)
            );
            assert_eq!(
                policy_json["provider_tpm_limits"]["openai"].as_u64(),
                Some(8800)
            );
            assert_eq!(
                policy_json["provider_tpm_limits"]["anthropic"].as_u64(),
                Some(6600)
            );
            assert_eq!(
                policy_json["provider_timeouts"]["openai"].as_u64(),
                Some(15)
            );
            assert_eq!(
                policy_json["provider_timeouts"]["anthropic"].as_u64(),
                Some(12)
            );
            assert_eq!(
                policy_json["provider_input_costs"]["openai"].as_f64(),
                Some(0.03)
            );
            assert_eq!(
                policy_json["provider_input_costs"]["anthropic"].as_f64(),
                Some(0.01)
            );
            assert_eq!(policy_json["semantic_cache_enabled"].as_bool(), Some(true));
            assert_eq!(policy_json["semantic_cache_ttl_secs"].as_u64(), Some(900));
            assert_eq!(
                policy_json["semantic_cache_similarity_threshold"].as_f64(),
                Some(0.72)
            );
        }

        #[tokio::test]
        async fn roles_and_principal_access_endpoints_report_effective_permissions() {
            let mgmt_token = "test-bootstrap-admin";
            let (_, api) = plugin_llm_gateway::create_plugins_with_options(
                &[],
                Some("sqlite::memory:"),
                &[],
                &[],
                CreatePluginsOptions {
                    bootstrap_admin_token: Some(mgmt_token.to_string()),
                    allow_direct_provider_keys: false,
                },
                None,
            )
            .await
            .unwrap();
            let port = start_test_server(api).await;

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/principals",
                r#"{"name":"alice"}"#,
                mgmt_token,
            )
            .await;
            assert_eq!(status, 201, "principal create failed: {body}");
            let principal_json: Value = serde_json::from_str(&body).unwrap();
            let principal_id = principal_json["principal_id"]
                .as_str()
                .expect("principal id");

            let (status, body) = post_with_bearer(
                port,
                "/api/v1/role-bindings",
                &serde_json::json!({
                    "principal_id": principal_id,
                    "role": "project_operator",
                    "project_id": "project-a",
                })
                .to_string(),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 201, "role binding create failed: {body}");

            let (status, body) = get_with_bearer(port, "/api/v1/roles", mgmt_token).await;
            assert_eq!(status, 200, "roles lookup failed: {body}");
            let roles_json: Value = serde_json::from_str(&body).unwrap();
            let project_operator = roles_json["roles"]
                .as_array()
                .expect("roles array")
                .iter()
                .find(|entry| entry["role"].as_str() == Some("project_operator"))
                .expect("project operator role");
            let permissions = project_operator["permissions"]
                .as_array()
                .expect("permissions array");
            assert!(permissions
                .iter()
                .any(|permission| permission.as_str() == Some("manage_runtime_keys")));
            assert!(permissions
                .iter()
                .any(|permission| permission.as_str() == Some("manage_project_policy")));
            assert!(!permissions
                .iter()
                .any(|permission| permission.as_str() == Some("manage_principals")));

            let (status, body) = get_with_bearer(
                port,
                &format!(
                    "/api/v1/principals/{}/access?project_id=project-a",
                    principal_id
                ),
                mgmt_token,
            )
            .await;
            assert_eq!(status, 200, "principal access lookup failed: {body}");
            let access_json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                access_json["principal"]["principal_id"].as_str(),
                Some(principal_id)
            );
            assert_eq!(
                access_json["scope"]["project_id"].as_str(),
                Some("project-a")
            );
            assert_eq!(
                access_json["project_access"][0]["project_id"].as_str(),
                Some("project-a")
            );
            let project_permissions = access_json["project_access"][0]["permissions"]
                .as_array()
                .expect("project permissions");
            assert!(project_permissions
                .iter()
                .any(|permission| permission.as_str() == Some("manage_runtime_keys")));
            assert!(project_permissions
                .iter()
                .any(|permission| permission.as_str() == Some("manage_project_policy")));
            assert!(!project_permissions
                .iter()
                .any(|permission| permission.as_str() == Some("manage_principals")));
            assert_eq!(
                access_json["project_access"][0]["role_bindings"][0]["role"].as_str(),
                Some("project_operator")
            );
        }
    }
}
