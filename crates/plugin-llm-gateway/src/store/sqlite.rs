use async_trait::async_trait;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use std::str::FromStr;

use super::schema::SQLITE_CREATE_TABLES;
use super::{
    GatewayStore, GovernanceChangeRecord, KeyModelUsageRecord, KeyUsageRecord,
    ManagedProviderRecord, ModelCostRecord, ProjectDatasetItemRecord, ProjectDatasetRecord,
    ProjectEvalRunItemRecord, ProjectEvalRunRecord, ProjectPolicyRecord, ProjectPromptRecord,
    ProjectPromptRolloutRecord, ProjectRolloutPolicyRecord, ProjectSemanticPolicyRecord,
    ProjectToolRecord, PromptCacheRouteRecord, RequestLogEntry, RequestLogQuery, RoutingRuleRecord,
    SafetyPolicyRecord, SemanticCacheEntryRecord, SessionEventRecord, SessionListQuery,
    SessionRecord, StoreError, VirtualKeyRecord,
};

pub struct SqliteStore {
    pool: SqlitePool,
}

fn map_request_log_row(r: sqlx::sqlite::SqliteRow) -> RequestLogEntry {
    RequestLogEntry {
        timestamp_unix: r.get::<i64, _>(0),
        api_key: r.get::<String, _>(1),
        project_id: r.get::<Option<String>, _>(2),
        session_id: r.get::<Option<String>, _>(3),
        metadata_json: r.get::<Option<String>, _>(4),
        custom_cost_json: r.get::<Option<String>, _>(5),
        custom_cost_applied: r.get::<i32, _>(6) != 0,
        provider_name: r.get::<Option<String>, _>(7),
        prompt_name: r.get::<Option<String>, _>(8),
        prompt_version: r.get::<Option<String>, _>(9),
        prompt_environment: r.get::<Option<String>, _>(10),
        model: r.get::<Option<String>, _>(11),
        input_tokens: r.get::<i64, _>(12) as u64,
        output_tokens: r.get::<i64, _>(13) as u64,
        cost: r.get::<f64, _>(14),
        is_streaming: r.get::<i32, _>(15) != 0,
        safety_mode: r.get::<Option<String>, _>(16),
        safety_matches: r.get::<Option<String>, _>(17),
        semantic_policy_version: r.get::<Option<String>, _>(18),
        semantic_index_state: r.get::<Option<String>, _>(19),
        semantic_degraded_reason: r.get::<Option<String>, _>(20),
        semantic_findings: r.get::<Option<String>, _>(21),
        tool_trace: r.get::<Option<String>, _>(22),
    }
}

impl SqliteStore {
    pub async fn connect(url: &str) -> Result<Self, StoreError> {
        let opts = SqliteConnectOptions::from_str(url)
            .map_err(|e| StoreError::Db(e.to_string()))?
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(opts)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        for sql in SQLITE_CREATE_TABLES {
            sqlx::query(sql)
                .execute(&pool)
                .await
                .map_err(|e| StoreError::Db(e.to_string()))?;
        }

        Ok(Self { pool })
    }
}

#[async_trait]
impl GatewayStore for SqliteStore {
    async fn get_usage(&self, api_key: &str) -> Result<Option<KeyUsageRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT total_input_tokens, total_output_tokens, total_cost FROM api_key_usage WHERE api_key = ?",
        )
        .bind(api_key)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| KeyUsageRecord {
            total_input_tokens: r.get::<i64, _>(0) as u64,
            total_output_tokens: r.get::<i64, _>(1) as u64,
            total_cost: r.get::<f64, _>(2),
        }))
    }

    async fn get_all_usage(&self) -> Result<Vec<(String, KeyUsageRecord)>, StoreError> {
        let rows = sqlx::query(
            "SELECT api_key, total_input_tokens, total_output_tokens, total_cost FROM api_key_usage",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| {
                (
                    r.get::<String, _>(0),
                    KeyUsageRecord {
                        total_input_tokens: r.get::<i64, _>(1) as u64,
                        total_output_tokens: r.get::<i64, _>(2) as u64,
                        total_cost: r.get::<f64, _>(3),
                    },
                )
            })
            .collect())
    }

    async fn upsert_usage(&self, api_key: &str, usage: &KeyUsageRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO api_key_usage (api_key, total_input_tokens, total_output_tokens, total_cost, updated_at)
             VALUES (?, ?, ?, ?, datetime('now'))",
        )
        .bind(api_key)
        .bind(usage.total_input_tokens as i64)
        .bind(usage.total_output_tokens as i64)
        .bind(usage.total_cost)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(())
    }

    async fn delete_usage(&self, api_key: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM api_key_usage WHERE api_key = ?")
            .bind(api_key)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn delete_all_usage(&self) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM api_key_usage")
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(())
    }

    async fn get_model_cost(&self, model: &str) -> Result<Option<ModelCostRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT input_cost_per_1k, output_cost_per_1k FROM model_pricing WHERE model_name = ?",
        )
        .bind(model)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ModelCostRecord {
            input_cost_per_1k: r.get::<f64, _>(0),
            output_cost_per_1k: r.get::<f64, _>(1),
        }))
    }

    async fn get_all_model_costs(&self) -> Result<Vec<(String, ModelCostRecord)>, StoreError> {
        let rows = sqlx::query(
            "SELECT model_name, input_cost_per_1k, output_cost_per_1k FROM model_pricing",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| {
                (
                    r.get::<String, _>(0),
                    ModelCostRecord {
                        input_cost_per_1k: r.get::<f64, _>(1),
                        output_cost_per_1k: r.get::<f64, _>(2),
                    },
                )
            })
            .collect())
    }

    async fn upsert_model_cost(
        &self,
        model: &str,
        cost: &ModelCostRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO model_pricing (model_name, input_cost_per_1k, output_cost_per_1k, updated_at)
             VALUES (?, ?, ?, datetime('now'))",
        )
        .bind(model)
        .bind(cost.input_cost_per_1k)
        .bind(cost.output_cost_per_1k)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(())
    }

    async fn delete_model_cost(&self, model: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM model_pricing WHERE model_name = ?")
            .bind(model)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn get_virtual_key(
        &self,
        key_hash: &str,
    ) -> Result<Option<VirtualKeyRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT key_hash, project_id, name, provider_name, budget_limit, budget_duration, \
             budget_window_start, rpm_limit, tpm_limit, allowed_models, timeout_secs, \
             tool_approval_mode, allowed_tools, active, created_at, expires_at \
             FROM virtual_keys WHERE key_hash = ?",
        )
        .bind(key_hash)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| VirtualKeyRecord {
            key_hash: r.get::<String, _>(0),
            project_id: r.get::<String, _>(1),
            name: r.get::<String, _>(2),
            provider_name: r.get::<String, _>(3),
            budget_limit: r.get::<Option<f64>, _>(4),
            budget_duration: r.get::<Option<String>, _>(5),
            budget_window_start: r.get::<Option<i64>, _>(6),
            rpm_limit: r.get::<Option<i32>, _>(7).map(|v| v as u32),
            tpm_limit: r.get::<Option<i32>, _>(8).map(|v| v as u32),
            allowed_models: r.get::<Option<String>, _>(9),
            timeout_secs: r.get::<Option<i64>, _>(10).map(|v| v as u64),
            tool_approval_mode: r.get::<Option<String>, _>(11),
            allowed_tools: r.get::<Option<String>, _>(12),
            active: r.get::<i32, _>(13) != 0,
            created_at: r.get::<String, _>(14),
            expires_at: r.get::<Option<String>, _>(15),
        }))
    }

    async fn get_all_virtual_keys(&self) -> Result<Vec<VirtualKeyRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT key_hash, project_id, name, provider_name, budget_limit, budget_duration, \
             budget_window_start, rpm_limit, tpm_limit, allowed_models, timeout_secs, \
             tool_approval_mode, allowed_tools, active, created_at, expires_at FROM virtual_keys",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| VirtualKeyRecord {
                key_hash: r.get::<String, _>(0),
                project_id: r.get::<String, _>(1),
                name: r.get::<String, _>(2),
                provider_name: r.get::<String, _>(3),
                budget_limit: r.get::<Option<f64>, _>(4),
                budget_duration: r.get::<Option<String>, _>(5),
                budget_window_start: r.get::<Option<i64>, _>(6),
                rpm_limit: r.get::<Option<i32>, _>(7).map(|v| v as u32),
                tpm_limit: r.get::<Option<i32>, _>(8).map(|v| v as u32),
                allowed_models: r.get::<Option<String>, _>(9),
                timeout_secs: r.get::<Option<i64>, _>(10).map(|v| v as u64),
                tool_approval_mode: r.get::<Option<String>, _>(11),
                allowed_tools: r.get::<Option<String>, _>(12),
                active: r.get::<i32, _>(13) != 0,
                created_at: r.get::<String, _>(14),
                expires_at: r.get::<Option<String>, _>(15),
            })
            .collect())
    }

    async fn upsert_virtual_key(&self, record: &VirtualKeyRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO virtual_keys \
             (key_hash, project_id, name, provider_name, budget_limit, budget_duration, \
              budget_window_start, rpm_limit, tpm_limit, allowed_models, timeout_secs, \
              tool_approval_mode, allowed_tools, active, created_at, expires_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.key_hash)
        .bind(&record.project_id)
        .bind(&record.name)
        .bind(&record.provider_name)
        .bind(record.budget_limit)
        .bind(&record.budget_duration)
        .bind(record.budget_window_start)
        .bind(record.rpm_limit.map(|v| v as i32))
        .bind(record.tpm_limit.map(|v| v as i32))
        .bind(&record.allowed_models)
        .bind(record.timeout_secs.map(|v| v as i64))
        .bind(&record.tool_approval_mode)
        .bind(&record.allowed_tools)
        .bind(if record.active { 1i32 } else { 0i32 })
        .bind(&record.created_at)
        .bind(&record.expires_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(())
    }

    async fn delete_virtual_key(&self, key_hash: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM virtual_keys WHERE key_hash = ?")
            .bind(key_hash)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(result.rows_affected() > 0)
    }

    async fn update_virtual_key_budget_window(
        &self,
        key_hash: &str,
        window_start: i64,
    ) -> Result<(), StoreError> {
        sqlx::query("UPDATE virtual_keys SET budget_window_start = ? WHERE key_hash = ?")
            .bind(window_start)
            .bind(key_hash)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(())
    }

    // --- Per-model usage ---

    async fn get_all_per_model_usage(&self) -> Result<Vec<KeyModelUsageRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT api_key, model, total_input_tokens, total_output_tokens, total_cost FROM api_key_model_usage",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| KeyModelUsageRecord {
                api_key: r.get::<String, _>(0),
                model: r.get::<String, _>(1),
                total_input_tokens: r.get::<i64, _>(2) as u64,
                total_output_tokens: r.get::<i64, _>(3) as u64,
                total_cost: r.get::<f64, _>(4),
            })
            .collect())
    }

    async fn upsert_per_model_usage(&self, record: &KeyModelUsageRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO api_key_model_usage \
             (api_key, model, total_input_tokens, total_output_tokens, total_cost, updated_at) \
             VALUES (?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&record.api_key)
        .bind(&record.model)
        .bind(record.total_input_tokens as i64)
        .bind(record.total_output_tokens as i64)
        .bind(record.total_cost)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(())
    }

    async fn delete_all_per_model_usage(&self) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM api_key_model_usage")
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(())
    }

    // --- Request log ---

    async fn append_request_logs(&self, entries: &[RequestLogEntry]) -> Result<(), StoreError> {
        for entry in entries {
            sqlx::query(
                "INSERT INTO request_log \
                 (timestamp_unix, api_key, project_id, session_id, metadata_json, custom_cost_json, custom_cost_applied, provider_name, prompt_name, prompt_version, prompt_environment, model, input_tokens, output_tokens, cost, is_streaming, safety_mode, safety_matches, semantic_policy_version, semantic_index_state, semantic_degraded_reason, semantic_findings, tool_trace) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(entry.timestamp_unix)
            .bind(&entry.api_key)
            .bind(&entry.project_id)
            .bind(&entry.session_id)
            .bind(&entry.metadata_json)
            .bind(&entry.custom_cost_json)
            .bind(if entry.custom_cost_applied { 1i32 } else { 0i32 })
            .bind(&entry.provider_name)
            .bind(&entry.prompt_name)
            .bind(&entry.prompt_version)
            .bind(&entry.prompt_environment)
            .bind(&entry.model)
            .bind(entry.input_tokens as i64)
            .bind(entry.output_tokens as i64)
            .bind(entry.cost)
            .bind(if entry.is_streaming { 1i32 } else { 0i32 })
            .bind(&entry.safety_mode)
            .bind(&entry.safety_matches)
            .bind(&entry.semantic_policy_version)
            .bind(&entry.semantic_index_state)
            .bind(&entry.semantic_degraded_reason)
            .bind(&entry.semantic_findings)
            .bind(&entry.tool_trace)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        }

        Ok(())
    }

    async fn get_request_logs(
        &self,
        api_key: Option<&str>,
        model: Option<&str>,
        project_id: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RequestLogEntry>, StoreError> {
        self.query_request_logs(&RequestLogQuery {
            api_key: api_key.map(ToString::to_string),
            model: model.map(ToString::to_string),
            project_id: project_id.map(ToString::to_string),
            session_id: None,
            metadata_key: None,
            metadata_value: None,
            has_custom_cost: None,
            custom_cost_applied: None,
            limit,
        })
        .await
    }

    async fn get_request_logs_for_session(
        &self,
        session_id: &str,
        project_id: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RequestLogEntry>, StoreError> {
        self.query_request_logs(&RequestLogQuery {
            api_key: None,
            model: None,
            project_id: project_id.map(ToString::to_string),
            session_id: Some(session_id.to_string()),
            metadata_key: None,
            metadata_value: None,
            has_custom_cost: None,
            custom_cost_applied: None,
            limit,
        })
        .await
    }

    async fn query_request_logs(
        &self,
        request: &RequestLogQuery,
    ) -> Result<Vec<RequestLogEntry>, StoreError> {
        let mut sql = String::from(
            "SELECT timestamp_unix, api_key, project_id, session_id, metadata_json, custom_cost_json, custom_cost_applied, provider_name, prompt_name, prompt_version, prompt_environment, model, input_tokens, output_tokens, cost, is_streaming, safety_mode, safety_matches, semantic_policy_version, semantic_index_state, semantic_degraded_reason, semantic_findings, tool_trace FROM request_log"
        );
        let mut conditions = Vec::new();
        if request.session_id.is_some() {
            conditions.push("session_id = ?");
        }
        if request.api_key.is_some() {
            conditions.push("api_key = ?");
        }
        if request.project_id.is_some() {
            conditions.push("project_id = ?");
        }
        if request.model.is_some() {
            conditions.push("model = ?");
        }
        if let Some(has_custom_cost) = request.has_custom_cost {
            conditions.push(if has_custom_cost {
                "custom_cost_json IS NOT NULL"
            } else {
                "custom_cost_json IS NULL"
            });
        }
        if request.custom_cost_applied.is_some() {
            conditions.push("custom_cost_applied = ?");
        }
        if request.metadata_key.is_some() {
            conditions.push("metadata_json IS NOT NULL");
            conditions.push("json_valid(metadata_json)");
            conditions.push("json_type(metadata_json, '$.' || ?) IS NOT NULL");
            if request.metadata_value.is_some() {
                conditions.push(
                    "CASE json_type(metadata_json, '$.' || ?) \
                     WHEN 'true' THEN 'true' \
                     WHEN 'false' THEN 'false' \
                     ELSE CAST(json_extract(metadata_json, '$.' || ?) AS TEXT) \
                     END = ?",
                );
            }
        }
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY timestamp_unix DESC LIMIT ?");

        let mut query = sqlx::query(&sql);
        if let Some(session_id) = request.session_id.as_deref() {
            query = query.bind(session_id);
        }
        if let Some(api_key) = request.api_key.as_deref() {
            query = query.bind(api_key);
        }
        if let Some(project_id) = request.project_id.as_deref() {
            query = query.bind(project_id);
        }
        if let Some(model) = request.model.as_deref() {
            query = query.bind(model);
        }
        if let Some(custom_cost_applied) = request.custom_cost_applied {
            query = query.bind(if custom_cost_applied { 1i32 } else { 0i32 });
        }
        if let Some(metadata_key) = request.metadata_key.as_deref() {
            query = query.bind(metadata_key);
            if let Some(metadata_value) = request.metadata_value.as_deref() {
                query = query.bind(metadata_key);
                query = query.bind(metadata_key);
                query = query.bind(metadata_value);
            }
        }
        query = query.bind(request.limit as i64);

        let rows = query
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows.into_iter().map(map_request_log_row).collect())
    }

    async fn get_session(&self, session_id: &str) -> Result<Option<SessionRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT session_id, project_id, project_ids_json, first_request_unix, last_request_unix, updated_at_unix, request_count, streaming_request_count, total_input_tokens, total_output_tokens, total_cost, providers_json, models_json, prompt_names_json, prompt_versions_json, tool_names_json, latest_request_json, safety_event_count, semantic_event_count, semantic_degraded_count, tool_call_count, tool_error_count, status, owner_id, owner_acquired_at_unix, last_transition_at_unix, last_transition_reason, last_heartbeat_unix, lease_expires_at_unix, cancel_requested_at_unix, cancel_requested_by, cancel_reason, handoff_target_owner_id, handoff_requested_at_unix, handoff_reason, state_json, metadata_json \
             FROM session_state WHERE session_id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| SessionRecord {
            session_id: r.get::<String, _>(0),
            project_id: r.get::<Option<String>, _>(1),
            project_ids_json: r.get::<Option<String>, _>(2),
            first_request_unix: r.get::<Option<i64>, _>(3),
            last_request_unix: r.get::<Option<i64>, _>(4),
            updated_at_unix: r.get::<i64, _>(5),
            request_count: r.get::<i64, _>(6) as u64,
            streaming_request_count: r.get::<i64, _>(7) as u64,
            total_input_tokens: r.get::<i64, _>(8) as u64,
            total_output_tokens: r.get::<i64, _>(9) as u64,
            total_cost: r.get::<f64, _>(10),
            providers_json: r.get::<Option<String>, _>(11),
            models_json: r.get::<Option<String>, _>(12),
            prompt_names_json: r.get::<Option<String>, _>(13),
            prompt_versions_json: r.get::<Option<String>, _>(14),
            tool_names_json: r.get::<Option<String>, _>(15),
            latest_request_json: r.get::<Option<String>, _>(16),
            safety_event_count: r.get::<i64, _>(17) as u64,
            semantic_event_count: r.get::<i64, _>(18) as u64,
            semantic_degraded_count: r.get::<i64, _>(19) as u64,
            tool_call_count: r.get::<i64, _>(20) as u64,
            tool_error_count: r.get::<i64, _>(21) as u64,
            status: r.get::<Option<String>, _>(22),
            owner_id: r.get::<Option<String>, _>(23),
            owner_acquired_at_unix: r.get::<Option<i64>, _>(24),
            last_transition_at_unix: r.get::<Option<i64>, _>(25),
            last_transition_reason: r.get::<Option<String>, _>(26),
            last_heartbeat_unix: r.get::<Option<i64>, _>(27),
            lease_expires_at_unix: r.get::<Option<i64>, _>(28),
            cancel_requested_at_unix: r.get::<Option<i64>, _>(29),
            cancel_requested_by: r.get::<Option<String>, _>(30),
            cancel_reason: r.get::<Option<String>, _>(31),
            handoff_target_owner_id: r.get::<Option<String>, _>(32),
            handoff_requested_at_unix: r.get::<Option<i64>, _>(33),
            handoff_reason: r.get::<Option<String>, _>(34),
            state_json: r.get::<Option<String>, _>(35),
            metadata_json: r.get::<Option<String>, _>(36),
        }))
    }

    async fn list_sessions(
        &self,
        query: &SessionListQuery,
    ) -> Result<Vec<SessionRecord>, StoreError> {
        let mut sql = "SELECT session_id, project_id, project_ids_json, first_request_unix, last_request_unix, updated_at_unix, request_count, streaming_request_count, total_input_tokens, total_output_tokens, total_cost, providers_json, models_json, prompt_names_json, prompt_versions_json, tool_names_json, latest_request_json, safety_event_count, semantic_event_count, semantic_degraded_count, tool_call_count, tool_error_count, status, owner_id, owner_acquired_at_unix, last_transition_at_unix, last_transition_reason, last_heartbeat_unix, lease_expires_at_unix, cancel_requested_at_unix, cancel_requested_by, cancel_reason, handoff_target_owner_id, handoff_requested_at_unix, handoff_reason, state_json, metadata_json FROM session_state WHERE 1=1".to_string();
        if query.project_id.is_some() {
            sql.push_str(" AND (project_id = ? OR project_ids_json LIKE ?)");
        }
        if query.status.is_some() {
            sql.push_str(" AND LOWER(status) = LOWER(?)");
        }
        if query.owner_id.is_some() {
            sql.push_str(" AND owner_id = ?");
        }
        if query.updated_after_unix.is_some() {
            sql.push_str(" AND updated_at_unix >= ?");
        }
        sql.push_str(
            " ORDER BY updated_at_unix DESC, COALESCE(last_request_unix, updated_at_unix) DESC LIMIT ?",
        );

        let mut db_query = sqlx::query(&sql);
        if let Some(project_id) = query.project_id.as_deref() {
            db_query = db_query
                .bind(project_id)
                .bind(format!("%\"{}\"%", project_id));
        }
        if let Some(status) = query.status.as_deref() {
            db_query = db_query.bind(status);
        }
        if let Some(owner_id) = query.owner_id.as_deref() {
            db_query = db_query.bind(owner_id);
        }
        if let Some(updated_after_unix) = query.updated_after_unix {
            db_query = db_query.bind(updated_after_unix);
        }
        let limit = if query.limit == 0 { 100 } else { query.limit };
        let rows = db_query
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| SessionRecord {
                session_id: r.get::<String, _>(0),
                project_id: r.get::<Option<String>, _>(1),
                project_ids_json: r.get::<Option<String>, _>(2),
                first_request_unix: r.get::<Option<i64>, _>(3),
                last_request_unix: r.get::<Option<i64>, _>(4),
                updated_at_unix: r.get::<i64, _>(5),
                request_count: r.get::<i64, _>(6) as u64,
                streaming_request_count: r.get::<i64, _>(7) as u64,
                total_input_tokens: r.get::<i64, _>(8) as u64,
                total_output_tokens: r.get::<i64, _>(9) as u64,
                total_cost: r.get::<f64, _>(10),
                providers_json: r.get::<Option<String>, _>(11),
                models_json: r.get::<Option<String>, _>(12),
                prompt_names_json: r.get::<Option<String>, _>(13),
                prompt_versions_json: r.get::<Option<String>, _>(14),
                tool_names_json: r.get::<Option<String>, _>(15),
                latest_request_json: r.get::<Option<String>, _>(16),
                safety_event_count: r.get::<i64, _>(17) as u64,
                semantic_event_count: r.get::<i64, _>(18) as u64,
                semantic_degraded_count: r.get::<i64, _>(19) as u64,
                tool_call_count: r.get::<i64, _>(20) as u64,
                tool_error_count: r.get::<i64, _>(21) as u64,
                status: r.get::<Option<String>, _>(22),
                owner_id: r.get::<Option<String>, _>(23),
                owner_acquired_at_unix: r.get::<Option<i64>, _>(24),
                last_transition_at_unix: r.get::<Option<i64>, _>(25),
                last_transition_reason: r.get::<Option<String>, _>(26),
                last_heartbeat_unix: r.get::<Option<i64>, _>(27),
                lease_expires_at_unix: r.get::<Option<i64>, _>(28),
                cancel_requested_at_unix: r.get::<Option<i64>, _>(29),
                cancel_requested_by: r.get::<Option<String>, _>(30),
                cancel_reason: r.get::<Option<String>, _>(31),
                handoff_target_owner_id: r.get::<Option<String>, _>(32),
                handoff_requested_at_unix: r.get::<Option<i64>, _>(33),
                handoff_reason: r.get::<Option<String>, _>(34),
                state_json: r.get::<Option<String>, _>(35),
                metadata_json: r.get::<Option<String>, _>(36),
            })
            .collect())
    }

    async fn list_sessions_for_recovery(
        &self,
        now_unix: i64,
        limit: u32,
    ) -> Result<Vec<String>, StoreError> {
        sqlx::query_scalar::<_, String>(
            "SELECT session_id FROM session_state \
             WHERE (status IS NULL OR LOWER(status) NOT IN ('completed', 'cancelled', 'failed')) \
               AND ( \
                 (owner_id IS NOT NULL AND (lease_expires_at_unix IS NULL OR lease_expires_at_unix <= ?)) \
                 OR (cancel_requested_at_unix IS NOT NULL AND (owner_id IS NULL OR lease_expires_at_unix IS NULL OR lease_expires_at_unix <= ?)) \
               ) \
             ORDER BY COALESCE(lease_expires_at_unix, cancel_requested_at_unix, updated_at_unix) ASC \
             LIMIT ?",
        )
        .bind(now_unix)
        .bind(now_unix)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))
    }

    async fn upsert_session(&self, record: &SessionRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO session_state \
             (session_id, project_id, project_ids_json, first_request_unix, last_request_unix, updated_at_unix, request_count, streaming_request_count, total_input_tokens, total_output_tokens, total_cost, providers_json, models_json, prompt_names_json, prompt_versions_json, tool_names_json, latest_request_json, safety_event_count, semantic_event_count, semantic_degraded_count, tool_call_count, tool_error_count, status, owner_id, owner_acquired_at_unix, last_transition_at_unix, last_transition_reason, last_heartbeat_unix, lease_expires_at_unix, cancel_requested_at_unix, cancel_requested_by, cancel_reason, handoff_target_owner_id, handoff_requested_at_unix, handoff_reason, state_json, metadata_json) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(session_id) DO UPDATE SET \
               project_id = excluded.project_id, \
               project_ids_json = excluded.project_ids_json, \
               first_request_unix = excluded.first_request_unix, \
               last_request_unix = excluded.last_request_unix, \
               updated_at_unix = excluded.updated_at_unix, \
               request_count = excluded.request_count, \
               streaming_request_count = excluded.streaming_request_count, \
               total_input_tokens = excluded.total_input_tokens, \
               total_output_tokens = excluded.total_output_tokens, \
               total_cost = excluded.total_cost, \
               providers_json = excluded.providers_json, \
               models_json = excluded.models_json, \
               prompt_names_json = excluded.prompt_names_json, \
               prompt_versions_json = excluded.prompt_versions_json, \
               tool_names_json = excluded.tool_names_json, \
               latest_request_json = excluded.latest_request_json, \
               safety_event_count = excluded.safety_event_count, \
               semantic_event_count = excluded.semantic_event_count, \
               semantic_degraded_count = excluded.semantic_degraded_count, \
               tool_call_count = excluded.tool_call_count, \
               tool_error_count = excluded.tool_error_count, \
               status = excluded.status, \
               owner_id = excluded.owner_id, \
               owner_acquired_at_unix = excluded.owner_acquired_at_unix, \
               last_transition_at_unix = excluded.last_transition_at_unix, \
               last_transition_reason = excluded.last_transition_reason, \
               last_heartbeat_unix = excluded.last_heartbeat_unix, \
               lease_expires_at_unix = excluded.lease_expires_at_unix, \
               cancel_requested_at_unix = excluded.cancel_requested_at_unix, \
               cancel_requested_by = excluded.cancel_requested_by, \
               cancel_reason = excluded.cancel_reason, \
               handoff_target_owner_id = excluded.handoff_target_owner_id, \
               handoff_requested_at_unix = excluded.handoff_requested_at_unix, \
               handoff_reason = excluded.handoff_reason, \
               state_json = excluded.state_json, \
               metadata_json = excluded.metadata_json",
        )
        .bind(&record.session_id)
        .bind(&record.project_id)
        .bind(&record.project_ids_json)
        .bind(record.first_request_unix)
        .bind(record.last_request_unix)
        .bind(record.updated_at_unix)
        .bind(record.request_count as i64)
        .bind(record.streaming_request_count as i64)
        .bind(record.total_input_tokens as i64)
        .bind(record.total_output_tokens as i64)
        .bind(record.total_cost)
        .bind(&record.providers_json)
        .bind(&record.models_json)
        .bind(&record.prompt_names_json)
        .bind(&record.prompt_versions_json)
        .bind(&record.tool_names_json)
        .bind(&record.latest_request_json)
        .bind(record.safety_event_count as i64)
        .bind(record.semantic_event_count as i64)
        .bind(record.semantic_degraded_count as i64)
        .bind(record.tool_call_count as i64)
        .bind(record.tool_error_count as i64)
        .bind(&record.status)
        .bind(&record.owner_id)
        .bind(record.owner_acquired_at_unix)
        .bind(record.last_transition_at_unix)
        .bind(&record.last_transition_reason)
        .bind(record.last_heartbeat_unix)
        .bind(record.lease_expires_at_unix)
        .bind(record.cancel_requested_at_unix)
        .bind(&record.cancel_requested_by)
        .bind(&record.cancel_reason)
        .bind(&record.handoff_target_owner_id)
        .bind(record.handoff_requested_at_unix)
        .bind(&record.handoff_reason)
        .bind(&record.state_json)
        .bind(&record.metadata_json)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(())
    }

    async fn append_session_event(&self, record: &SessionEventRecord) -> Result<i64, StoreError> {
        let result = sqlx::query(
            "INSERT INTO session_event \
             (session_id, project_id, event_kind, actor_id, reason, payload_json, created_at_unix) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.session_id)
        .bind(&record.project_id)
        .bind(&record.event_kind)
        .bind(&record.actor_id)
        .bind(&record.reason)
        .bind(&record.payload_json)
        .bind(record.created_at_unix)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(result.last_insert_rowid())
    }

    async fn get_session_events(
        &self,
        session_id: &str,
        after_seq: Option<i64>,
        limit: u32,
    ) -> Result<Vec<SessionEventRecord>, StoreError> {
        let mut sql = "SELECT event_seq, session_id, project_id, event_kind, actor_id, reason, payload_json, created_at_unix \
             FROM session_event WHERE session_id = ?"
            .to_string();
        if after_seq.is_some() {
            sql.push_str(" AND event_seq > ?");
        }
        sql.push_str(" ORDER BY event_seq ASC LIMIT ?");

        let mut query = sqlx::query(&sql).bind(session_id);
        if let Some(after_seq) = after_seq {
            query = query.bind(after_seq);
        }
        let rows = query
            .bind(if limit == 0 { 100_i64 } else { limit as i64 })
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| SessionEventRecord {
                event_seq: r.get::<i64, _>(0),
                session_id: r.get::<String, _>(1),
                project_id: r.get::<Option<String>, _>(2),
                event_kind: r.get::<String, _>(3),
                actor_id: r.get::<Option<String>, _>(4),
                reason: r.get::<Option<String>, _>(5),
                payload_json: r.get::<Option<String>, _>(6),
                created_at_unix: r.get::<i64, _>(7),
            })
            .collect())
    }

    async fn get_project_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<ProjectPolicyRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, budget_limit, budget_duration, rpm_limit, tpm_limit, fallback_order, adaptive_enabled, timeout_secs, provider_rpm_limits, provider_tpm_limits, provider_timeouts, provider_input_costs, provider_output_costs, semantic_cache_enabled, semantic_cache_ttl_secs, semantic_cache_similarity_threshold, tool_approval_mode, allowed_tools, updated_at \
             FROM project_policies WHERE project_id = ?",
        )
        .bind(project_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectPolicyRecord {
            project_id: r.get::<String, _>(0),
            budget_limit: r.get::<Option<f64>, _>(1),
            budget_duration: r.get::<Option<String>, _>(2),
            rpm_limit: r.get::<Option<i32>, _>(3).map(|v| v as u32),
            tpm_limit: r.get::<Option<i32>, _>(4).map(|v| v as u32),
            fallback_order: r.get::<Option<String>, _>(5),
            adaptive_enabled: r.get::<i32, _>(6) != 0,
            timeout_secs: r.get::<Option<i64>, _>(7).map(|v| v as u64),
            provider_rpm_limits: r.get::<Option<String>, _>(8),
            provider_tpm_limits: r.get::<Option<String>, _>(9),
            provider_timeouts: r.get::<Option<String>, _>(10),
            provider_input_costs: r.get::<Option<String>, _>(11),
            provider_output_costs: r.get::<Option<String>, _>(12),
            semantic_cache_enabled: r.get::<Option<i32>, _>(13).map(|v| v != 0),
            semantic_cache_ttl_secs: r.get::<Option<i64>, _>(14).map(|v| v as u64),
            semantic_cache_similarity_threshold: r.get::<Option<f64>, _>(15),
            tool_approval_mode: r.get::<Option<String>, _>(16),
            allowed_tools: r.get::<Option<String>, _>(17),
            updated_at: r.get::<String, _>(18),
        }))
    }

    async fn get_all_project_policies(&self) -> Result<Vec<ProjectPolicyRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT project_id, budget_limit, budget_duration, rpm_limit, tpm_limit, fallback_order, adaptive_enabled, timeout_secs, provider_rpm_limits, provider_tpm_limits, provider_timeouts, provider_input_costs, provider_output_costs, semantic_cache_enabled, semantic_cache_ttl_secs, semantic_cache_similarity_threshold, tool_approval_mode, allowed_tools, updated_at FROM project_policies",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectPolicyRecord {
                project_id: r.get::<String, _>(0),
                budget_limit: r.get::<Option<f64>, _>(1),
                budget_duration: r.get::<Option<String>, _>(2),
                rpm_limit: r.get::<Option<i32>, _>(3).map(|v| v as u32),
                tpm_limit: r.get::<Option<i32>, _>(4).map(|v| v as u32),
                fallback_order: r.get::<Option<String>, _>(5),
                adaptive_enabled: r.get::<i32, _>(6) != 0,
                timeout_secs: r.get::<Option<i64>, _>(7).map(|v| v as u64),
                provider_rpm_limits: r.get::<Option<String>, _>(8),
                provider_tpm_limits: r.get::<Option<String>, _>(9),
                provider_timeouts: r.get::<Option<String>, _>(10),
                provider_input_costs: r.get::<Option<String>, _>(11),
                provider_output_costs: r.get::<Option<String>, _>(12),
                semantic_cache_enabled: r.get::<Option<i32>, _>(13).map(|v| v != 0),
                semantic_cache_ttl_secs: r.get::<Option<i64>, _>(14).map(|v| v as u64),
                semantic_cache_similarity_threshold: r.get::<Option<f64>, _>(15),
                tool_approval_mode: r.get::<Option<String>, _>(16),
                allowed_tools: r.get::<Option<String>, _>(17),
                updated_at: r.get::<String, _>(18),
            })
            .collect())
    }

    async fn upsert_project_policy(&self, record: &ProjectPolicyRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_policies \
             (project_id, budget_limit, budget_duration, rpm_limit, tpm_limit, fallback_order, adaptive_enabled, timeout_secs, provider_rpm_limits, provider_tpm_limits, provider_timeouts, provider_input_costs, provider_output_costs, semantic_cache_enabled, semantic_cache_ttl_secs, semantic_cache_similarity_threshold, tool_approval_mode, allowed_tools, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&record.project_id)
        .bind(record.budget_limit)
        .bind(&record.budget_duration)
        .bind(record.rpm_limit.map(|v| v as i32))
        .bind(record.tpm_limit.map(|v| v as i32))
        .bind(&record.fallback_order)
        .bind(if record.adaptive_enabled { 1i32 } else { 0i32 })
        .bind(record.timeout_secs.map(|v| v as i64))
        .bind(&record.provider_rpm_limits)
        .bind(&record.provider_tpm_limits)
        .bind(&record.provider_timeouts)
        .bind(&record.provider_input_costs)
        .bind(&record.provider_output_costs)
        .bind(record.semantic_cache_enabled.map(|value| if value { 1i32 } else { 0i32 }))
        .bind(record.semantic_cache_ttl_secs.map(|v| v as i64))
        .bind(record.semantic_cache_similarity_threshold)
        .bind(&record.tool_approval_mode)
        .bind(&record.allowed_tools)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_project_policy(&self, project_id: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM project_policies WHERE project_id = ?")
            .bind(project_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_managed_provider(
        &self,
        name: &str,
    ) -> Result<Option<ManagedProviderRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT name, enabled, api_key_env, base_url, models_json, api_key_header, timeout_secs, family, surfaces_json, routing_metadata_json, created_at, updated_at \
             FROM managed_providers WHERE name = ?",
        )
        .bind(name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ManagedProviderRecord {
            name: r.get::<String, _>(0),
            enabled: r.get::<i32, _>(1) != 0,
            api_key_env: r.get::<Option<String>, _>(2),
            base_url: r.get::<Option<String>, _>(3),
            models_json: r.get::<Option<String>, _>(4),
            api_key_header: r.get::<Option<String>, _>(5),
            timeout_secs: r.get::<Option<i64>, _>(6).map(|value| value as u64),
            family: r.get::<Option<String>, _>(7),
            surfaces_json: r.get::<Option<String>, _>(8),
            routing_metadata_json: r.get::<Option<String>, _>(9),
            created_at: r.get::<String, _>(10),
            updated_at: r.get::<String, _>(11),
        }))
    }

    async fn get_managed_providers(&self) -> Result<Vec<ManagedProviderRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT name, enabled, api_key_env, base_url, models_json, api_key_header, timeout_secs, family, surfaces_json, routing_metadata_json, created_at, updated_at \
             FROM managed_providers ORDER BY name ASC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ManagedProviderRecord {
                name: r.get::<String, _>(0),
                enabled: r.get::<i32, _>(1) != 0,
                api_key_env: r.get::<Option<String>, _>(2),
                base_url: r.get::<Option<String>, _>(3),
                models_json: r.get::<Option<String>, _>(4),
                api_key_header: r.get::<Option<String>, _>(5),
                timeout_secs: r.get::<Option<i64>, _>(6).map(|value| value as u64),
                family: r.get::<Option<String>, _>(7),
                surfaces_json: r.get::<Option<String>, _>(8),
                routing_metadata_json: r.get::<Option<String>, _>(9),
                created_at: r.get::<String, _>(10),
                updated_at: r.get::<String, _>(11),
            })
            .collect())
    }

    async fn upsert_managed_provider(
        &self,
        record: &ManagedProviderRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO managed_providers \
             (name, enabled, api_key_env, base_url, models_json, api_key_header, timeout_secs, family, surfaces_json, routing_metadata_json, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.name)
        .bind(if record.enabled { 1i32 } else { 0i32 })
        .bind(&record.api_key_env)
        .bind(&record.base_url)
        .bind(&record.models_json)
        .bind(&record.api_key_header)
        .bind(record.timeout_secs.map(|value| value as i64))
        .bind(&record.family)
        .bind(&record.surfaces_json)
        .bind(&record.routing_metadata_json)
        .bind(&record.created_at)
        .bind(&record.updated_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_managed_provider(&self, name: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM managed_providers WHERE name = ?")
            .bind(name)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_routing_rules(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<RoutingRuleRecord>, StoreError> {
        let rows = if let Some(project_id) = project_id {
            sqlx::query(
                "SELECT rule_id, project_id, name, priority, enabled, match_path, match_model, match_streaming, match_role, match_headers, min_prompt_tokens, max_prompt_tokens, deny_reason, provider_order, provider_weights, timeout_secs, created_at \
                 FROM routing_rules WHERE project_id = ? ORDER BY priority DESC, created_at ASC",
            )
            .bind(project_id)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query(
                "SELECT rule_id, project_id, name, priority, enabled, match_path, match_model, match_streaming, match_role, match_headers, min_prompt_tokens, max_prompt_tokens, deny_reason, provider_order, provider_weights, timeout_secs, created_at \
                 FROM routing_rules ORDER BY priority DESC, created_at ASC",
            )
            .fetch_all(&self.pool)
            .await
        }
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| RoutingRuleRecord {
                rule_id: r.get::<String, _>(0),
                project_id: r.get::<String, _>(1),
                name: r.get::<String, _>(2),
                priority: r.get::<i32, _>(3),
                enabled: r.get::<i32, _>(4) != 0,
                match_path: r.get::<Option<String>, _>(5),
                match_model: r.get::<Option<String>, _>(6),
                match_streaming: r.get::<Option<i32>, _>(7).map(|v| v != 0),
                match_role: r.get::<Option<String>, _>(8),
                match_headers: r.get::<Option<String>, _>(9),
                min_prompt_tokens: r.get::<Option<i32>, _>(10).map(|v| v as u32),
                max_prompt_tokens: r.get::<Option<i32>, _>(11).map(|v| v as u32),
                deny_reason: r.get::<Option<String>, _>(12),
                provider_order: r.get::<Option<String>, _>(13),
                provider_weights: r.get::<Option<String>, _>(14),
                timeout_secs: r.get::<Option<i64>, _>(15).map(|v| v as u64),
                created_at: r.get::<String, _>(16),
            })
            .collect())
    }

    async fn upsert_routing_rule(&self, record: &RoutingRuleRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO routing_rules \
             (rule_id, project_id, name, priority, enabled, match_path, match_model, match_streaming, match_role, match_headers, min_prompt_tokens, max_prompt_tokens, deny_reason, provider_order, provider_weights, timeout_secs, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.rule_id)
        .bind(&record.project_id)
        .bind(&record.name)
        .bind(record.priority)
        .bind(if record.enabled { 1i32 } else { 0i32 })
        .bind(&record.match_path)
        .bind(&record.match_model)
        .bind(record.match_streaming.map(|v| if v { 1i32 } else { 0i32 }))
        .bind(&record.match_role)
        .bind(&record.match_headers)
        .bind(record.min_prompt_tokens.map(|v| v as i32))
        .bind(record.max_prompt_tokens.map(|v| v as i32))
        .bind(&record.deny_reason)
        .bind(&record.provider_order)
        .bind(&record.provider_weights)
        .bind(record.timeout_secs.map(|v| v as i64))
        .bind(&record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_routing_rule(&self, rule_id: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM routing_rules WHERE rule_id = ?")
            .bind(rule_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_safety_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<SafetyPolicyRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, mode, rules_json, updated_at FROM safety_policies WHERE project_id = ?",
        )
        .bind(project_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| SafetyPolicyRecord {
            project_id: r.get::<String, _>(0),
            mode: r.get::<String, _>(1),
            rules_json: r.get::<Option<String>, _>(2),
            updated_at: r.get::<String, _>(3),
        }))
    }

    async fn get_all_safety_policies(&self) -> Result<Vec<SafetyPolicyRecord>, StoreError> {
        let rows =
            sqlx::query("SELECT project_id, mode, rules_json, updated_at FROM safety_policies")
                .fetch_all(&self.pool)
                .await
                .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| SafetyPolicyRecord {
                project_id: r.get::<String, _>(0),
                mode: r.get::<String, _>(1),
                rules_json: r.get::<Option<String>, _>(2),
                updated_at: r.get::<String, _>(3),
            })
            .collect())
    }

    async fn upsert_safety_policy(&self, record: &SafetyPolicyRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO safety_policies (project_id, mode, rules_json, updated_at) VALUES (?, ?, ?, datetime('now'))",
        )
        .bind(&record.project_id)
        .bind(&record.mode)
        .bind(&record.rules_json)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_safety_policy(&self, project_id: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM safety_policies WHERE project_id = ?")
            .bind(project_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_semantic_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<ProjectSemanticPolicyRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, version, enabled, entities_json, topics_json, updated_at FROM semantic_policies WHERE project_id = ?",
        )
        .bind(project_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectSemanticPolicyRecord {
            project_id: r.get::<String, _>(0),
            version: r.get::<String, _>(1),
            enabled: r.get::<i32, _>(2) != 0,
            entities_json: r.get::<Option<String>, _>(3),
            topics_json: r.get::<Option<String>, _>(4),
            updated_at: r.get::<String, _>(5),
        }))
    }

    async fn get_all_semantic_policies(
        &self,
    ) -> Result<Vec<ProjectSemanticPolicyRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT project_id, version, enabled, entities_json, topics_json, updated_at FROM semantic_policies",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectSemanticPolicyRecord {
                project_id: r.get::<String, _>(0),
                version: r.get::<String, _>(1),
                enabled: r.get::<i32, _>(2) != 0,
                entities_json: r.get::<Option<String>, _>(3),
                topics_json: r.get::<Option<String>, _>(4),
                updated_at: r.get::<String, _>(5),
            })
            .collect())
    }

    async fn upsert_semantic_policy(
        &self,
        record: &ProjectSemanticPolicyRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO semantic_policies (project_id, version, enabled, entities_json, topics_json, updated_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&record.project_id)
        .bind(&record.version)
        .bind(if record.enabled { 1i32 } else { 0i32 })
        .bind(&record.entities_json)
        .bind(&record.topics_json)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_semantic_policy(&self, project_id: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM semantic_policies WHERE project_id = ?")
            .bind(project_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_prompt_cache_routes(
        &self,
        now_ms: u64,
        limit: u32,
    ) -> Result<Vec<PromptCacheRouteRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT route_id, project_id, cache_key, provider_name, signal_kind, signal_strength,
                    observed_at_ms, expires_at_ms
             FROM prompt_cache_routes
             WHERE expires_at_ms > ?
             ORDER BY expires_at_ms DESC
             LIMIT ?",
        )
        .bind(now_ms as i64)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| PromptCacheRouteRecord {
                route_id: r.get::<String, _>(0),
                project_id: r.get::<String, _>(1),
                cache_key: r.get::<String, _>(2),
                provider_name: r.get::<String, _>(3),
                signal_kind: r.get::<String, _>(4),
                signal_strength: r.get::<f64, _>(5),
                observed_at_ms: r.get::<i64, _>(6) as u64,
                expires_at_ms: r.get::<i64, _>(7) as u64,
            })
            .collect())
    }

    async fn upsert_prompt_cache_route(
        &self,
        record: &PromptCacheRouteRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO prompt_cache_routes
                (route_id, project_id, cache_key, provider_name, signal_kind, signal_strength,
                 observed_at_ms, expires_at_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.route_id)
        .bind(&record.project_id)
        .bind(&record.cache_key)
        .bind(&record.provider_name)
        .bind(&record.signal_kind)
        .bind(record.signal_strength)
        .bind(record.observed_at_ms as i64)
        .bind(record.expires_at_ms as i64)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_prompt_cache_route(&self, route_id: &str) -> Result<bool, StoreError> {
        let result = sqlx::query("DELETE FROM prompt_cache_routes WHERE route_id = ?")
            .bind(route_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn prune_prompt_cache_routes(&self, now_ms: u64) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM prompt_cache_routes WHERE expires_at_ms <= ?")
            .bind(now_ms as i64)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn get_semantic_cache_entries(
        &self,
        now_ms: u64,
        limit: u32,
    ) -> Result<Vec<SemanticCacheEntryRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT cache_id, project_id, provider_name, model, tokens_json, response_status,
                    content_type, response_body, prompt_tokens, created_at_ms, expires_at_ms
             FROM semantic_cache_entries
             WHERE expires_at_ms > ?
             ORDER BY created_at_ms DESC
             LIMIT ?",
        )
        .bind(now_ms as i64)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| SemanticCacheEntryRecord {
                cache_id: r.get::<String, _>(0),
                project_id: r.get::<String, _>(1),
                provider_name: r.get::<String, _>(2),
                model: r.get::<String, _>(3),
                tokens_json: r.get::<String, _>(4),
                response_status: r.get::<i64, _>(5) as u16,
                content_type: r.get::<Option<String>, _>(6),
                response_body: r.get::<Vec<u8>, _>(7),
                prompt_tokens: r.get::<i64, _>(8) as u64,
                created_at_ms: r.get::<i64, _>(9) as u64,
                expires_at_ms: r.get::<i64, _>(10) as u64,
            })
            .collect())
    }

    async fn upsert_semantic_cache_entry(
        &self,
        record: &SemanticCacheEntryRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO semantic_cache_entries
                (cache_id, project_id, provider_name, model, tokens_json, response_status,
                 content_type, response_body, prompt_tokens, created_at_ms, expires_at_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.cache_id)
        .bind(&record.project_id)
        .bind(&record.provider_name)
        .bind(&record.model)
        .bind(&record.tokens_json)
        .bind(record.response_status as i64)
        .bind(&record.content_type)
        .bind(&record.response_body)
        .bind(record.prompt_tokens as i64)
        .bind(record.created_at_ms as i64)
        .bind(record.expires_at_ms as i64)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn prune_semantic_cache_entries(
        &self,
        now_ms: u64,
        max_entries: u32,
    ) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM semantic_cache_entries WHERE expires_at_ms <= ?")
            .bind(now_ms as i64)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        if max_entries == 0 {
            sqlx::query("DELETE FROM semantic_cache_entries")
                .execute(&self.pool)
                .await
                .map_err(|e| StoreError::Db(e.to_string()))?;
            return Ok(());
        }

        let total = sqlx::query("SELECT COUNT(*) FROM semantic_cache_entries")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?
            .get::<i64, _>(0) as u64;
        let max_entries = u64::from(max_entries);
        if total <= max_entries {
            return Ok(());
        }
        let remove_count = (total - max_entries) as i64;
        sqlx::query(
            "DELETE FROM semantic_cache_entries
             WHERE cache_id IN (
                 SELECT cache_id
                 FROM semantic_cache_entries
                 ORDER BY created_at_ms ASC
                 LIMIT ?
             )",
        )
        .bind(remove_count)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn get_project_tool(
        &self,
        project_id: &str,
        tool_name: &str,
    ) -> Result<Option<ProjectToolRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, tool_name, description, input_schema_json, executor_kind, executor_config_json, enabled, timeout_ms, updated_at
             FROM project_tools
             WHERE project_id = ? AND tool_name = ?",
        )
        .bind(project_id)
        .bind(tool_name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectToolRecord {
            project_id: r.get::<String, _>(0),
            tool_name: r.get::<String, _>(1),
            description: r.get::<Option<String>, _>(2),
            input_schema_json: r.get::<String, _>(3),
            executor_kind: r.get::<String, _>(4),
            executor_config_json: r.get::<Option<String>, _>(5),
            enabled: r.get::<i32, _>(6) != 0,
            timeout_ms: r.get::<Option<i64>, _>(7).map(|value| value as u64),
            updated_at: r.get::<String, _>(8),
        }))
    }

    async fn get_project_tools(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectToolRecord>, StoreError> {
        let rows = match project_id {
            Some(project_id) => {
                sqlx::query(
                    "SELECT project_id, tool_name, description, input_schema_json, executor_kind, executor_config_json, enabled, timeout_ms, updated_at
                     FROM project_tools
                     WHERE project_id = ?
                     ORDER BY tool_name",
                )
                .bind(project_id)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query(
                    "SELECT project_id, tool_name, description, input_schema_json, executor_kind, executor_config_json, enabled, timeout_ms, updated_at
                     FROM project_tools
                     ORDER BY project_id, tool_name",
                )
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectToolRecord {
                project_id: r.get::<String, _>(0),
                tool_name: r.get::<String, _>(1),
                description: r.get::<Option<String>, _>(2),
                input_schema_json: r.get::<String, _>(3),
                executor_kind: r.get::<String, _>(4),
                executor_config_json: r.get::<Option<String>, _>(5),
                enabled: r.get::<i32, _>(6) != 0,
                timeout_ms: r.get::<Option<i64>, _>(7).map(|value| value as u64),
                updated_at: r.get::<String, _>(8),
            })
            .collect())
    }

    async fn upsert_project_tool(&self, record: &ProjectToolRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_tools (project_id, tool_name, description, input_schema_json, executor_kind, executor_config_json, enabled, timeout_ms, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&record.project_id)
        .bind(&record.tool_name)
        .bind(&record.description)
        .bind(&record.input_schema_json)
        .bind(&record.executor_kind)
        .bind(&record.executor_config_json)
        .bind(if record.enabled { 1i32 } else { 0i32 })
        .bind(record.timeout_ms.map(|value| value as i64))
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_project_tool(
        &self,
        project_id: &str,
        tool_name: &str,
    ) -> Result<bool, StoreError> {
        let result =
            sqlx::query("DELETE FROM project_tools WHERE project_id = ? AND tool_name = ?")
                .bind(project_id)
                .bind(tool_name)
                .execute(&self.pool)
                .await
                .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Result<Option<ProjectPromptRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, prompt_name, version, environment, description, target, template_text, variables_schema_json, rollout_metadata_json, active, updated_at
             FROM project_prompts
             WHERE project_id = ? AND prompt_name = ? AND version = ?",
        )
        .bind(project_id)
        .bind(prompt_name)
        .bind(version)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectPromptRecord {
            project_id: r.get::<String, _>(0),
            prompt_name: r.get::<String, _>(1),
            version: r.get::<String, _>(2),
            environment: r.get::<String, _>(3),
            description: r.get::<Option<String>, _>(4),
            target: r.get::<String, _>(5),
            template_text: r.get::<String, _>(6),
            variables_schema_json: r.get::<Option<String>, _>(7),
            rollout_metadata_json: r.get::<Option<String>, _>(8),
            active: r.get::<i32, _>(9) != 0,
            updated_at: r.get::<String, _>(10),
        }))
    }

    async fn get_project_prompts(
        &self,
        project_id: Option<&str>,
        prompt_name: Option<&str>,
    ) -> Result<Vec<ProjectPromptRecord>, StoreError> {
        let rows = match (project_id, prompt_name) {
            (Some(project_id), Some(prompt_name)) => {
                sqlx::query(
                    "SELECT project_id, prompt_name, version, environment, description, target, template_text, variables_schema_json, rollout_metadata_json, active, updated_at
                     FROM project_prompts
                     WHERE project_id = ? AND prompt_name = ?
                     ORDER BY prompt_name, environment, version",
                )
                .bind(project_id)
                .bind(prompt_name)
                .fetch_all(&self.pool)
                .await
            }
            (Some(project_id), None) => {
                sqlx::query(
                    "SELECT project_id, prompt_name, version, environment, description, target, template_text, variables_schema_json, rollout_metadata_json, active, updated_at
                     FROM project_prompts
                     WHERE project_id = ?
                     ORDER BY prompt_name, environment, version",
                )
                .bind(project_id)
                .fetch_all(&self.pool)
                .await
            }
            (None, Some(prompt_name)) => {
                sqlx::query(
                    "SELECT project_id, prompt_name, version, environment, description, target, template_text, variables_schema_json, rollout_metadata_json, active, updated_at
                     FROM project_prompts
                     WHERE prompt_name = ?
                     ORDER BY project_id, prompt_name, environment, version",
                )
                .bind(prompt_name)
                .fetch_all(&self.pool)
                .await
            }
            (None, None) => {
                sqlx::query(
                    "SELECT project_id, prompt_name, version, environment, description, target, template_text, variables_schema_json, rollout_metadata_json, active, updated_at
                     FROM project_prompts
                     ORDER BY project_id, prompt_name, environment, version",
                )
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectPromptRecord {
                project_id: r.get::<String, _>(0),
                prompt_name: r.get::<String, _>(1),
                version: r.get::<String, _>(2),
                environment: r.get::<String, _>(3),
                description: r.get::<Option<String>, _>(4),
                target: r.get::<String, _>(5),
                template_text: r.get::<String, _>(6),
                variables_schema_json: r.get::<Option<String>, _>(7),
                rollout_metadata_json: r.get::<Option<String>, _>(8),
                active: r.get::<i32, _>(9) != 0,
                updated_at: r.get::<String, _>(10),
            })
            .collect())
    }

    async fn upsert_project_prompt(&self, record: &ProjectPromptRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_prompts (project_id, prompt_name, version, environment, description, target, template_text, variables_schema_json, rollout_metadata_json, active, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&record.project_id)
        .bind(&record.prompt_name)
        .bind(&record.version)
        .bind(&record.environment)
        .bind(&record.description)
        .bind(&record.target)
        .bind(&record.template_text)
        .bind(&record.variables_schema_json)
        .bind(&record.rollout_metadata_json)
        .bind(if record.active { 1i32 } else { 0i32 })
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Result<bool, StoreError> {
        let result = sqlx::query(
            "DELETE FROM project_prompts WHERE project_id = ? AND prompt_name = ? AND version = ?",
        )
        .bind(project_id)
        .bind(prompt_name)
        .bind(version)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_project_rollout_policy(
        &self,
        project_id: &str,
        policy_name: &str,
    ) -> Result<Option<ProjectRolloutPolicyRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, policy_name, description, gate_config_json, target_environment, updated_at
             FROM project_rollout_policies
             WHERE project_id = ? AND policy_name = ?",
        )
        .bind(project_id)
        .bind(policy_name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectRolloutPolicyRecord {
            project_id: r.get::<String, _>(0),
            policy_name: r.get::<String, _>(1),
            description: r.get::<Option<String>, _>(2),
            gate_config_json: r.get::<String, _>(3),
            target_environment: r.get::<Option<String>, _>(4),
            updated_at: r.get::<String, _>(5),
        }))
    }

    async fn get_project_rollout_policies(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectRolloutPolicyRecord>, StoreError> {
        let rows = match project_id {
            Some(project_id) => {
                sqlx::query(
                    "SELECT project_id, policy_name, description, gate_config_json, target_environment, updated_at
                     FROM project_rollout_policies
                     WHERE project_id = ?
                     ORDER BY policy_name",
                )
                .bind(project_id)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query(
                    "SELECT project_id, policy_name, description, gate_config_json, target_environment, updated_at
                     FROM project_rollout_policies
                     ORDER BY project_id, policy_name",
                )
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectRolloutPolicyRecord {
                project_id: r.get::<String, _>(0),
                policy_name: r.get::<String, _>(1),
                description: r.get::<Option<String>, _>(2),
                gate_config_json: r.get::<String, _>(3),
                target_environment: r.get::<Option<String>, _>(4),
                updated_at: r.get::<String, _>(5),
            })
            .collect())
    }

    async fn upsert_project_rollout_policy(
        &self,
        record: &ProjectRolloutPolicyRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_rollout_policies (project_id, policy_name, description, gate_config_json, target_environment, updated_at)
             VALUES (?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&record.project_id)
        .bind(&record.policy_name)
        .bind(&record.description)
        .bind(&record.gate_config_json)
        .bind(&record.target_environment)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_project_rollout_policy(
        &self,
        project_id: &str,
        policy_name: &str,
    ) -> Result<bool, StoreError> {
        let result = sqlx::query(
            "DELETE FROM project_rollout_policies WHERE project_id = ? AND policy_name = ?",
        )
        .bind(project_id)
        .bind(policy_name)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_project_prompt_rollout(
        &self,
        project_id: &str,
        prompt_name: &str,
        rollout_id: &str,
    ) -> Result<Option<ProjectPromptRolloutRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, prompt_name, rollout_id, policy_name, baseline_version, candidate_version, baseline_run_id, candidate_run_id, target_environment, status, recommendation_action, comparison_json, created_at, applied_at
             FROM project_prompt_rollouts
             WHERE project_id = ? AND prompt_name = ? AND rollout_id = ?",
        )
        .bind(project_id)
        .bind(prompt_name)
        .bind(rollout_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectPromptRolloutRecord {
            project_id: r.get::<String, _>(0),
            prompt_name: r.get::<String, _>(1),
            rollout_id: r.get::<String, _>(2),
            policy_name: r.get::<String, _>(3),
            baseline_version: r.get::<Option<String>, _>(4),
            candidate_version: r.get::<String, _>(5),
            baseline_run_id: r.get::<String, _>(6),
            candidate_run_id: r.get::<String, _>(7),
            target_environment: r.get::<Option<String>, _>(8),
            status: r.get::<String, _>(9),
            recommendation_action: r.get::<Option<String>, _>(10),
            comparison_json: r.get::<String, _>(11),
            created_at: r.get::<String, _>(12),
            applied_at: r.get::<Option<String>, _>(13),
        }))
    }

    async fn get_project_prompt_rollouts(
        &self,
        project_id: &str,
        prompt_name: &str,
    ) -> Result<Vec<ProjectPromptRolloutRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT project_id, prompt_name, rollout_id, policy_name, baseline_version, candidate_version, baseline_run_id, candidate_run_id, target_environment, status, recommendation_action, comparison_json, created_at, applied_at
             FROM project_prompt_rollouts
             WHERE project_id = ? AND prompt_name = ?
             ORDER BY created_at DESC, rollout_id DESC",
        )
        .bind(project_id)
        .bind(prompt_name)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectPromptRolloutRecord {
                project_id: r.get::<String, _>(0),
                prompt_name: r.get::<String, _>(1),
                rollout_id: r.get::<String, _>(2),
                policy_name: r.get::<String, _>(3),
                baseline_version: r.get::<Option<String>, _>(4),
                candidate_version: r.get::<String, _>(5),
                baseline_run_id: r.get::<String, _>(6),
                candidate_run_id: r.get::<String, _>(7),
                target_environment: r.get::<Option<String>, _>(8),
                status: r.get::<String, _>(9),
                recommendation_action: r.get::<Option<String>, _>(10),
                comparison_json: r.get::<String, _>(11),
                created_at: r.get::<String, _>(12),
                applied_at: r.get::<Option<String>, _>(13),
            })
            .collect())
    }

    async fn upsert_project_prompt_rollout(
        &self,
        record: &ProjectPromptRolloutRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_prompt_rollouts (project_id, prompt_name, rollout_id, policy_name, baseline_version, candidate_version, baseline_run_id, candidate_run_id, target_environment, status, recommendation_action, comparison_json, created_at, applied_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.project_id)
        .bind(&record.prompt_name)
        .bind(&record.rollout_id)
        .bind(&record.policy_name)
        .bind(&record.baseline_version)
        .bind(&record.candidate_version)
        .bind(&record.baseline_run_id)
        .bind(&record.candidate_run_id)
        .bind(&record.target_environment)
        .bind(&record.status)
        .bind(&record.recommendation_action)
        .bind(&record.comparison_json)
        .bind(&record.created_at)
        .bind(&record.applied_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn get_project_dataset(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<Option<ProjectDatasetRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, dataset_name, description, schema_json, updated_at
             FROM project_datasets
             WHERE project_id = ? AND dataset_name = ?",
        )
        .bind(project_id)
        .bind(dataset_name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectDatasetRecord {
            project_id: r.get::<String, _>(0),
            dataset_name: r.get::<String, _>(1),
            description: r.get::<Option<String>, _>(2),
            schema_json: r.get::<Option<String>, _>(3),
            updated_at: r.get::<String, _>(4),
        }))
    }

    async fn get_project_datasets(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectDatasetRecord>, StoreError> {
        let rows = match project_id {
            Some(project_id) => {
                sqlx::query(
                    "SELECT project_id, dataset_name, description, schema_json, updated_at
                     FROM project_datasets
                     WHERE project_id = ?
                     ORDER BY dataset_name",
                )
                .bind(project_id)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query(
                    "SELECT project_id, dataset_name, description, schema_json, updated_at
                     FROM project_datasets
                     ORDER BY project_id, dataset_name",
                )
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectDatasetRecord {
                project_id: r.get::<String, _>(0),
                dataset_name: r.get::<String, _>(1),
                description: r.get::<Option<String>, _>(2),
                schema_json: r.get::<Option<String>, _>(3),
                updated_at: r.get::<String, _>(4),
            })
            .collect())
    }

    async fn upsert_project_dataset(
        &self,
        record: &ProjectDatasetRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_datasets (project_id, dataset_name, description, schema_json, updated_at)
             VALUES (?, ?, ?, ?, datetime('now'))",
        )
        .bind(&record.project_id)
        .bind(&record.dataset_name)
        .bind(&record.description)
        .bind(&record.schema_json)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_project_dataset(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<bool, StoreError> {
        sqlx::query("DELETE FROM project_dataset_items WHERE project_id = ? AND dataset_name = ?")
            .bind(project_id)
            .bind(dataset_name)
            .execute(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;
        let result =
            sqlx::query("DELETE FROM project_datasets WHERE project_id = ? AND dataset_name = ?")
                .bind(project_id)
                .bind(dataset_name)
                .execute(&self.pool)
                .await
                .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_project_dataset_item(
        &self,
        project_id: &str,
        dataset_name: &str,
        item_id: &str,
    ) -> Result<Option<ProjectDatasetItemRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, dataset_name, item_id, input_json, expected_output_json, metadata_json, updated_at
             FROM project_dataset_items
             WHERE project_id = ? AND dataset_name = ? AND item_id = ?",
        )
        .bind(project_id)
        .bind(dataset_name)
        .bind(item_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectDatasetItemRecord {
            project_id: r.get::<String, _>(0),
            dataset_name: r.get::<String, _>(1),
            item_id: r.get::<String, _>(2),
            input_json: r.get::<String, _>(3),
            expected_output_json: r.get::<Option<String>, _>(4),
            metadata_json: r.get::<Option<String>, _>(5),
            updated_at: r.get::<String, _>(6),
        }))
    }

    async fn get_project_dataset_items(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<Vec<ProjectDatasetItemRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT project_id, dataset_name, item_id, input_json, expected_output_json, metadata_json, updated_at
             FROM project_dataset_items
             WHERE project_id = ? AND dataset_name = ?
             ORDER BY item_id",
        )
        .bind(project_id)
        .bind(dataset_name)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectDatasetItemRecord {
                project_id: r.get::<String, _>(0),
                dataset_name: r.get::<String, _>(1),
                item_id: r.get::<String, _>(2),
                input_json: r.get::<String, _>(3),
                expected_output_json: r.get::<Option<String>, _>(4),
                metadata_json: r.get::<Option<String>, _>(5),
                updated_at: r.get::<String, _>(6),
            })
            .collect())
    }

    async fn upsert_project_dataset_item(
        &self,
        record: &ProjectDatasetItemRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_dataset_items (project_id, dataset_name, item_id, input_json, expected_output_json, metadata_json, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(&record.project_id)
        .bind(&record.dataset_name)
        .bind(&record.item_id)
        .bind(&record.input_json)
        .bind(&record.expected_output_json)
        .bind(&record.metadata_json)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn delete_project_dataset_item(
        &self,
        project_id: &str,
        dataset_name: &str,
        item_id: &str,
    ) -> Result<bool, StoreError> {
        let result = sqlx::query(
            "DELETE FROM project_dataset_items WHERE project_id = ? AND dataset_name = ? AND item_id = ?",
        )
        .bind(project_id)
        .bind(dataset_name)
        .bind(item_id)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_project_eval_run(
        &self,
        project_id: &str,
        run_id: &str,
    ) -> Result<Option<ProjectEvalRunRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT run_id, project_id, dataset_name, target_url, status, total_items, passed_items, failed_items,
                    total_input_tokens, total_output_tokens, total_cost, average_latency_ms, summary_json, created_at, completed_at
             FROM project_eval_runs
             WHERE project_id = ? AND run_id = ?",
        )
        .bind(project_id)
        .bind(run_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(row.map(|r| ProjectEvalRunRecord {
            run_id: r.get::<String, _>(0),
            project_id: r.get::<String, _>(1),
            dataset_name: r.get::<String, _>(2),
            target_url: r.get::<String, _>(3),
            status: r.get::<String, _>(4),
            total_items: r.get::<i64, _>(5) as u32,
            passed_items: r.get::<i64, _>(6) as u32,
            failed_items: r.get::<i64, _>(7) as u32,
            total_input_tokens: r.get::<i64, _>(8) as u64,
            total_output_tokens: r.get::<i64, _>(9) as u64,
            total_cost: r.get::<f64, _>(10),
            average_latency_ms: r.get::<f64, _>(11),
            summary_json: r.get::<Option<String>, _>(12),
            created_at: r.get::<String, _>(13),
            completed_at: r.get::<Option<String>, _>(14),
        }))
    }

    async fn get_project_eval_runs(
        &self,
        project_id: &str,
        dataset_name: Option<&str>,
    ) -> Result<Vec<ProjectEvalRunRecord>, StoreError> {
        let rows = match dataset_name {
            Some(dataset_name) => {
                sqlx::query(
                    "SELECT run_id, project_id, dataset_name, target_url, status, total_items, passed_items, failed_items,
                            total_input_tokens, total_output_tokens, total_cost, average_latency_ms, summary_json, created_at, completed_at
                     FROM project_eval_runs
                     WHERE project_id = ? AND dataset_name = ?
                     ORDER BY created_at DESC, run_id DESC",
                )
                .bind(project_id)
                .bind(dataset_name)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query(
                    "SELECT run_id, project_id, dataset_name, target_url, status, total_items, passed_items, failed_items,
                            total_input_tokens, total_output_tokens, total_cost, average_latency_ms, summary_json, created_at, completed_at
                     FROM project_eval_runs
                     WHERE project_id = ?
                     ORDER BY created_at DESC, run_id DESC",
                )
                .bind(project_id)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectEvalRunRecord {
                run_id: r.get::<String, _>(0),
                project_id: r.get::<String, _>(1),
                dataset_name: r.get::<String, _>(2),
                target_url: r.get::<String, _>(3),
                status: r.get::<String, _>(4),
                total_items: r.get::<i64, _>(5) as u32,
                passed_items: r.get::<i64, _>(6) as u32,
                failed_items: r.get::<i64, _>(7) as u32,
                total_input_tokens: r.get::<i64, _>(8) as u64,
                total_output_tokens: r.get::<i64, _>(9) as u64,
                total_cost: r.get::<f64, _>(10),
                average_latency_ms: r.get::<f64, _>(11),
                summary_json: r.get::<Option<String>, _>(12),
                created_at: r.get::<String, _>(13),
                completed_at: r.get::<Option<String>, _>(14),
            })
            .collect())
    }

    async fn get_project_eval_runs_by_status(
        &self,
        statuses: &[&str],
        limit: u32,
    ) -> Result<Vec<ProjectEvalRunRecord>, StoreError> {
        if statuses.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = vec!["?"; statuses.len()].join(", ");
        let sql = format!(
            "SELECT run_id, project_id, dataset_name, target_url, status, total_items, passed_items, failed_items,
                    total_input_tokens, total_output_tokens, total_cost, average_latency_ms, summary_json, created_at, completed_at
             FROM project_eval_runs
             WHERE status IN ({placeholders})
             ORDER BY created_at ASC, run_id ASC
             LIMIT ?"
        );
        let mut query = sqlx::query(&sql);
        for status in statuses {
            query = query.bind(status);
        }
        let rows = query
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectEvalRunRecord {
                run_id: r.get::<String, _>(0),
                project_id: r.get::<String, _>(1),
                dataset_name: r.get::<String, _>(2),
                target_url: r.get::<String, _>(3),
                status: r.get::<String, _>(4),
                total_items: r.get::<i64, _>(5) as u32,
                passed_items: r.get::<i64, _>(6) as u32,
                failed_items: r.get::<i64, _>(7) as u32,
                total_input_tokens: r.get::<i64, _>(8) as u64,
                total_output_tokens: r.get::<i64, _>(9) as u64,
                total_cost: r.get::<f64, _>(10),
                average_latency_ms: r.get::<f64, _>(11),
                summary_json: r.get::<Option<String>, _>(12),
                created_at: r.get::<String, _>(13),
                completed_at: r.get::<Option<String>, _>(14),
            })
            .collect())
    }

    async fn upsert_project_eval_run(
        &self,
        record: &ProjectEvalRunRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_eval_runs
             (run_id, project_id, dataset_name, target_url, status, total_items, passed_items, failed_items,
              total_input_tokens, total_output_tokens, total_cost, average_latency_ms, summary_json, created_at, completed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.run_id)
        .bind(&record.project_id)
        .bind(&record.dataset_name)
        .bind(&record.target_url)
        .bind(&record.status)
        .bind(record.total_items as i64)
        .bind(record.passed_items as i64)
        .bind(record.failed_items as i64)
        .bind(record.total_input_tokens as i64)
        .bind(record.total_output_tokens as i64)
        .bind(record.total_cost)
        .bind(record.average_latency_ms)
        .bind(&record.summary_json)
        .bind(&record.created_at)
        .bind(&record.completed_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn get_project_eval_run_items(
        &self,
        project_id: &str,
        run_id: &str,
    ) -> Result<Vec<ProjectEvalRunItemRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT run_id, project_id, dataset_name, item_id, passed, status_code, latency_ms, output_text,
                    evaluation_json, error, input_tokens, output_tokens, cost, created_at
             FROM project_eval_run_items
             WHERE project_id = ? AND run_id = ?
             ORDER BY item_id",
        )
        .bind(project_id)
        .bind(run_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| ProjectEvalRunItemRecord {
                run_id: r.get::<String, _>(0),
                project_id: r.get::<String, _>(1),
                dataset_name: r.get::<String, _>(2),
                item_id: r.get::<String, _>(3),
                passed: r.get::<i64, _>(4) != 0,
                status_code: r.get::<Option<i64>, _>(5).map(|value| value as u16),
                latency_ms: r.get::<i64, _>(6) as u64,
                output_text: r.get::<Option<String>, _>(7),
                evaluation_json: r.get::<Option<String>, _>(8),
                error: r.get::<Option<String>, _>(9),
                input_tokens: r.get::<i64, _>(10) as u64,
                output_tokens: r.get::<i64, _>(11) as u64,
                cost: r.get::<f64, _>(12),
                created_at: r.get::<String, _>(13),
            })
            .collect())
    }

    async fn upsert_project_eval_run_item(
        &self,
        record: &ProjectEvalRunItemRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO project_eval_run_items
             (run_id, project_id, dataset_name, item_id, passed, status_code, latency_ms, output_text,
              evaluation_json, error, input_tokens, output_tokens, cost, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.run_id)
        .bind(&record.project_id)
        .bind(&record.dataset_name)
        .bind(&record.item_id)
        .bind(if record.passed { 1_i64 } else { 0_i64 })
        .bind(record.status_code.map(i64::from))
        .bind(record.latency_ms as i64)
        .bind(&record.output_text)
        .bind(&record.evaluation_json)
        .bind(&record.error)
        .bind(record.input_tokens as i64)
        .bind(record.output_tokens as i64)
        .bind(record.cost)
        .bind(&record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn append_governance_change(
        &self,
        record: &GovernanceChangeRecord,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR REPLACE INTO governance_history
             (change_id, project_id, resource_type, resource_id, action, before_json, after_json, changed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.change_id)
        .bind(&record.project_id)
        .bind(&record.resource_type)
        .bind(&record.resource_id)
        .bind(&record.action)
        .bind(&record.before_json)
        .bind(&record.after_json)
        .bind(&record.changed_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Db(e.to_string()))?;
        Ok(())
    }

    async fn get_governance_changes(
        &self,
        project_id: &str,
        resource_type: Option<&str>,
        limit: u32,
    ) -> Result<Vec<GovernanceChangeRecord>, StoreError> {
        let rows = match resource_type {
            Some(resource_type) => {
                sqlx::query(
                    "SELECT change_id, project_id, resource_type, resource_id, action, before_json, after_json, changed_at
                     FROM governance_history
                     WHERE project_id = ? AND resource_type = ?
                     ORDER BY changed_at DESC, change_id DESC
                     LIMIT ?",
                )
                .bind(project_id)
                .bind(resource_type)
                .bind(limit as i64)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query(
                    "SELECT change_id, project_id, resource_type, resource_id, action, before_json, after_json, changed_at
                     FROM governance_history
                     WHERE project_id = ?
                     ORDER BY changed_at DESC, change_id DESC
                     LIMIT ?",
                )
                .bind(project_id)
                .bind(limit as i64)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|row| GovernanceChangeRecord {
                change_id: row.get::<String, _>(0),
                project_id: row.get::<String, _>(1),
                resource_type: row.get::<String, _>(2),
                resource_id: row.get::<String, _>(3),
                action: row.get::<String, _>(4),
                before_json: row.get::<Option<String>, _>(5),
                after_json: row.get::<Option<String>, _>(6),
                changed_at: row.get::<String, _>(7),
            })
            .collect())
    }
}
