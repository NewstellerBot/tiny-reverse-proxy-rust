use plugin_llm_gateway::store::{
    self, GatewayStore, KeyModelUsageRecord, KeyUsageRecord, ModelCostRecord, ProjectPolicyRecord,
    RequestLogEntry, RoutingRuleRecord, SafetyPolicyRecord, VirtualKeyRecord,
};

fn unique_id(prefix: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    format!("{prefix}-{nanos}")
}

async fn run_gateway_round_trip(url: &str) {
    let store = store::connect(url).await.expect("connect store");
    let suffix = unique_id("rt");

    let usage_key = format!("usage-{suffix}");
    store
        .upsert_usage(
            &usage_key,
            &KeyUsageRecord {
                total_input_tokens: 123,
                total_output_tokens: 45,
                total_cost: 0.67,
            },
        )
        .await
        .expect("upsert usage");
    let usage = store
        .get_usage(&usage_key)
        .await
        .expect("get usage")
        .expect("usage row");
    assert_eq!(usage.total_input_tokens, 123);
    assert_eq!(usage.total_output_tokens, 45);

    let model = format!("model-{suffix}");
    store
        .upsert_model_cost(
            &model,
            &ModelCostRecord {
                input_cost_per_1k: 0.11,
                output_cost_per_1k: 0.22,
            },
        )
        .await
        .expect("upsert model cost");
    let model_cost = store
        .get_model_cost(&model)
        .await
        .expect("get model cost")
        .expect("model cost");
    assert!((model_cost.input_cost_per_1k - 0.11).abs() < 1e-9);

    store
        .upsert_per_model_usage(&KeyModelUsageRecord {
            api_key: usage_key.clone(),
            model: model.clone(),
            total_input_tokens: 555,
            total_output_tokens: 444,
            total_cost: 0.33,
        })
        .await
        .expect("upsert per-model usage");
    let per_model = store
        .get_all_per_model_usage()
        .await
        .expect("get per-model usage");
    assert!(per_model.iter().any(|entry| {
        entry.api_key == usage_key && entry.model == model && entry.total_input_tokens == 555
    }));

    let project_id = format!("project-{suffix}");
    let key_hash = format!("hash-{suffix}");
    store
        .upsert_virtual_key(&VirtualKeyRecord {
            key_hash: key_hash.clone(),
            project_id: project_id.clone(),
            name: format!("key-{suffix}"),
            provider_name: "openai".to_string(),
            budget_limit: Some(12.5),
            budget_duration: Some("daily".to_string()),
            budget_window_start: Some(100),
            rpm_limit: Some(10),
            tpm_limit: Some(1000),
            allowed_models: Some("[\"gpt-4o\"]".to_string()),
            timeout_secs: None,
            tool_approval_mode: Some("allow_list".to_string()),
            allowed_tools: Some("[\"web_search\"]".to_string()),
            active: true,
            created_at: "111".to_string(),
            expires_at: Some("222".to_string()),
        })
        .await
        .expect("upsert virtual key");
    store
        .update_virtual_key_budget_window(&key_hash, 999)
        .await
        .expect("update budget window");
    let virtual_key = store
        .get_virtual_key(&key_hash)
        .await
        .expect("get virtual key")
        .expect("virtual key");
    assert_eq!(virtual_key.project_id, project_id);
    assert_eq!(virtual_key.budget_window_start, Some(999));
    assert_eq!(virtual_key.expires_at.as_deref(), Some("222"));

    store
        .upsert_project_policy(&ProjectPolicyRecord {
            project_id: project_id.clone(),
            budget_limit: Some(99.0),
            budget_duration: Some("monthly".to_string()),
            rpm_limit: Some(77),
            tpm_limit: Some(7777),
            fallback_order: Some("[\"openai\",\"anthropic\"]".to_string()),
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
            allowed_tools: Some("[\"web_search\"]".to_string()),
            updated_at: "333".to_string(),
        })
        .await
        .expect("upsert project policy");
    let project_policy = store
        .get_project_policy(&project_id)
        .await
        .expect("get project policy")
        .expect("project policy");
    assert_eq!(project_policy.rpm_limit, Some(77));
    assert!(project_policy.adaptive_enabled);
    assert_eq!(project_policy.semantic_cache_enabled, None);

    let rule_id = format!("rule-{suffix}");
    store
        .upsert_routing_rule(&RoutingRuleRecord {
            rule_id: rule_id.clone(),
            project_id: project_id.clone(),
            name: "prefer-openai".to_string(),
            priority: 10,
            enabled: true,
            match_path: Some("/v1/chat/completions".to_string()),
            match_model: Some("gpt-4o".to_string()),
            match_streaming: Some(true),
            match_role: Some("project_runtime".to_string()),
            match_headers: Some("{\"x-test\":\"1\"}".to_string()),
            min_prompt_tokens: Some(10),
            max_prompt_tokens: Some(1000),
            deny_reason: None,
            provider_order: Some("[\"openai\",\"anthropic\"]".to_string()),
            provider_weights: Some("{\"openai\":80,\"anthropic\":20}".to_string()),
            timeout_secs: Some(30),
            created_at: "444".to_string(),
        })
        .await
        .expect("upsert routing rule");
    let rules = store
        .get_routing_rules(Some(&project_id))
        .await
        .expect("get routing rules");
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].rule_id, rule_id);
    assert_eq!(rules[0].match_streaming, Some(true));

    store
        .upsert_safety_policy(&SafetyPolicyRecord {
            project_id: project_id.clone(),
            mode: "redact_and_forward".to_string(),
            rules_json: Some(
                "[{\"pattern\":\"sk-[a-z0-9]+\",\"description\":\"secret\"}]".to_string(),
            ),
            updated_at: "555".to_string(),
        })
        .await
        .expect("upsert safety policy");
    let safety = store
        .get_safety_policy(&project_id)
        .await
        .expect("get safety policy")
        .expect("safety policy");
    assert_eq!(safety.mode, "redact_and_forward");

    store
        .append_request_logs(&[
            RequestLogEntry {
                timestamp_unix: 1000,
                api_key: usage_key.clone(),
                project_id: Some(project_id.clone()),
                session_id: Some("session-db".to_string()),
                metadata_json: None,
                custom_cost_json: None,
                custom_cost_applied: false,
                provider_name: Some("openai".to_string()),
                prompt_name: Some("support".to_string()),
                prompt_version: Some("v1".to_string()),
                prompt_environment: Some("prod".to_string()),
                model: Some(model.clone()),
                input_tokens: 10,
                output_tokens: 20,
                cost: 0.01,
                is_streaming: false,
                safety_mode: Some("observe_only".to_string()),
                safety_matches: Some("[\"secret\"]".to_string()),
                semantic_policy_version: Some("sem-v1".to_string()),
                semantic_index_state: Some("ready".to_string()),
                semantic_degraded_reason: None,
                semantic_findings: Some("[]".to_string()),
                tool_trace: None,
            },
            RequestLogEntry {
                timestamp_unix: 2000,
                api_key: usage_key.clone(),
                project_id: Some(project_id.clone()),
                session_id: Some("session-db".to_string()),
                metadata_json: None,
                custom_cost_json: None,
                custom_cost_applied: false,
                provider_name: Some("anthropic".to_string()),
                prompt_name: Some("analysis".to_string()),
                prompt_version: Some("v2".to_string()),
                prompt_environment: Some("staging".to_string()),
                model: Some(model.clone()),
                input_tokens: 30,
                output_tokens: 40,
                cost: 0.02,
                is_streaming: true,
                safety_mode: Some("redact_and_forward".to_string()),
                safety_matches: Some("[\"pii\"]".to_string()),
                semantic_policy_version: Some("sem-v2".to_string()),
                semantic_index_state: Some("degraded".to_string()),
                semantic_degraded_reason: Some("semantic safety timeout".to_string()),
                semantic_findings: Some(
                    r#"[{"chunk_path":"$.input","topic_id":"layoffs"}]"#.to_string(),
                ),
                tool_trace: Some(
                    r#"{"calls":[{"tool_name":"arxiv_search","executor_kind":"arxiv_search","status":"success","error_code":null}]}"#
                        .to_string(),
                ),
            },
        ])
        .await
        .expect("append request logs");
    let logs = store
        .get_request_logs(Some(&usage_key), Some(&model), Some(&project_id), 10)
        .await
        .expect("get request logs");
    assert_eq!(logs.len(), 2);
    assert_eq!(logs[0].timestamp_unix, 2000);
    assert!(logs[0].is_streaming);
    assert_eq!(logs[0].prompt_name.as_deref(), Some("analysis"));
    assert_eq!(logs[0].prompt_version.as_deref(), Some("v2"));
    assert_eq!(logs[0].prompt_environment.as_deref(), Some("staging"));
    assert_eq!(logs[0].semantic_index_state.as_deref(), Some("degraded"));
    assert_eq!(logs[0].session_id.as_deref(), Some("session-db"));
    assert_eq!(logs[1].prompt_name.as_deref(), Some("support"));
    assert_eq!(logs[1].safety_mode.as_deref(), Some("observe_only"));

    let session_logs = store
        .get_request_logs_for_session("session-db", Some(&project_id), 10)
        .await
        .expect("get request logs for session");
    assert_eq!(session_logs.len(), 2);
    assert_eq!(
        session_logs[0].tool_trace.as_deref(),
        Some(
            r#"{"calls":[{"tool_name":"arxiv_search","executor_kind":"arxiv_search","status":"success","error_code":null}]}"#
        )
    );
}

#[tokio::test]
#[ignore = "requires TRP_TEST_POSTGRES_URL"]
async fn postgres_gateway_round_trip() {
    let url = std::env::var("TRP_TEST_POSTGRES_URL").expect("TRP_TEST_POSTGRES_URL");
    run_gateway_round_trip(&url).await;
}

#[tokio::test]
#[ignore = "requires TRP_TEST_MYSQL_URL"]
async fn mysql_gateway_round_trip() {
    let url = std::env::var("TRP_TEST_MYSQL_URL").expect("TRP_TEST_MYSQL_URL");
    run_gateway_round_trip(&url).await;
}
