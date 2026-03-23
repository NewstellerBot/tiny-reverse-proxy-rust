use async_trait::async_trait;
use sqlx::mysql::MySqlPoolOptions;
use sqlx::{MySql, MySqlPool, QueryBuilder, Row};

use super::schema::MYSQL_CREATE_TABLES;
use super::{
    GatewayStore, GovernanceChangeRecord, KeyModelUsageRecord, KeyUsageRecord,
    ManagedProviderRecord, ModelCostRecord, ProjectDatasetItemRecord, ProjectDatasetRecord,
    ProjectEvalRunItemRecord, ProjectEvalRunRecord, ProjectPolicyRecord, ProjectPromptRecord,
    ProjectPromptRolloutRecord, ProjectRolloutPolicyRecord, ProjectSemanticPolicyRecord,
    ProjectToolRecord, PromptCacheRouteRecord, RequestLogEntry, RequestLogQuery, RoutingRuleRecord,
    SafetyPolicyRecord, SemanticCacheEntryRecord, SessionEventRecord, SessionListQuery,
    SessionRecord, StoreError, VirtualKeyRecord,
};

pub struct MysqlStore {
    pool: MySqlPool,
}

impl MysqlStore {
    pub async fn connect(url: &str) -> Result<Self, StoreError> {
        let pool = MySqlPoolOptions::new()
            .max_connections(4)
            .connect(url)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        for sql in MYSQL_CREATE_TABLES {
            sqlx::query(sql)
                .execute(&pool)
                .await
                .map_err(|e| StoreError::Db(e.to_string()))?;
        }

        Ok(Self { pool })
    }
}

#[async_trait]
impl GatewayStore for MysqlStore {
    async fn get_usage(&self, api_key: &str) -> Result<Option<KeyUsageRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT total_input_tokens, total_output_tokens, total_cost
             FROM api_key_usage
             WHERE api_key = ?",
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
            "INSERT INTO api_key_usage
                 (api_key, total_input_tokens, total_output_tokens, total_cost, updated_at)
             VALUES
                 (?, ?, ?, ?, CAST(UNIX_TIMESTAMP() AS CHAR))
             ON DUPLICATE KEY UPDATE
                 total_input_tokens = VALUES(total_input_tokens),
                 total_output_tokens = VALUES(total_output_tokens),
                 total_cost = VALUES(total_cost),
                 updated_at = VALUES(updated_at)",
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
            "SELECT input_cost_per_1k, output_cost_per_1k
             FROM model_pricing
             WHERE model_name = ?",
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
            "INSERT INTO model_pricing
                 (model_name, input_cost_per_1k, output_cost_per_1k, updated_at)
             VALUES
                 (?, ?, ?, CAST(UNIX_TIMESTAMP() AS CHAR))
             ON DUPLICATE KEY UPDATE
                 input_cost_per_1k = VALUES(input_cost_per_1k),
                 output_cost_per_1k = VALUES(output_cost_per_1k),
                 updated_at = VALUES(updated_at)",
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

    async fn get_all_per_model_usage(&self) -> Result<Vec<KeyModelUsageRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT api_key, model, total_input_tokens, total_output_tokens, total_cost
             FROM api_key_model_usage",
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
            "INSERT INTO api_key_model_usage
                 (api_key, model, total_input_tokens, total_output_tokens, total_cost, updated_at)
             VALUES
                 (?, ?, ?, ?, ?, CAST(UNIX_TIMESTAMP() AS CHAR))
             ON DUPLICATE KEY UPDATE
                 total_input_tokens = VALUES(total_input_tokens),
                 total_output_tokens = VALUES(total_output_tokens),
                 total_cost = VALUES(total_cost),
                 updated_at = VALUES(updated_at)",
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

    async fn append_request_logs(&self, entries: &[RequestLogEntry]) -> Result<(), StoreError> {
        for entry in entries {
            sqlx::query(
                "INSERT INTO request_log
                     (timestamp_unix, api_key, project_id, session_id, metadata_json, custom_cost_json, custom_cost_applied, provider_name, prompt_name, prompt_version,
                      prompt_environment, model, input_tokens, output_tokens, cost, is_streaming,
                      safety_mode, safety_matches, semantic_policy_version, semantic_index_state,
                      semantic_degraded_reason, semantic_findings, tool_trace)
                 VALUES
                     (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(entry.timestamp_unix)
            .bind(&entry.api_key)
            .bind(&entry.project_id)
            .bind(&entry.session_id)
            .bind(&entry.metadata_json)
            .bind(&entry.custom_cost_json)
            .bind(entry.custom_cost_applied)
            .bind(&entry.provider_name)
            .bind(&entry.prompt_name)
            .bind(&entry.prompt_version)
            .bind(&entry.prompt_environment)
            .bind(&entry.model)
            .bind(entry.input_tokens as i64)
            .bind(entry.output_tokens as i64)
            .bind(entry.cost)
            .bind(entry.is_streaming)
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
        let mut query = QueryBuilder::<MySql>::new(
            "SELECT timestamp_unix, api_key, project_id, session_id, metadata_json, custom_cost_json, custom_cost_applied, provider_name, prompt_name, prompt_version,
                    prompt_environment, model, input_tokens, output_tokens, cost, is_streaming,
                    safety_mode, safety_matches, semantic_policy_version, semantic_index_state,
                    semantic_degraded_reason, semantic_findings, tool_trace
             FROM request_log",
        );

        let mut has_where = false;
        if let Some(session_id) = request.session_id.as_deref() {
            query.push(" WHERE session_id = ").push_bind(session_id);
            has_where = true;
        }
        if let Some(api_key) = request.api_key.as_deref() {
            query.push(if has_where { " AND " } else { " WHERE " });
            query.push("api_key = ").push_bind(api_key);
            has_where = true;
        }
        if let Some(project_id) = request.project_id.as_deref() {
            query.push(if has_where { " AND " } else { " WHERE " });
            query.push("project_id = ").push_bind(project_id);
            has_where = true;
        }
        if let Some(model) = request.model.as_deref() {
            query.push(if has_where { " AND " } else { " WHERE " });
            query.push("model = ").push_bind(model);
            has_where = true;
        }
        if let Some(has_custom_cost) = request.has_custom_cost {
            query.push(if has_where { " AND " } else { " WHERE " });
            query.push(if has_custom_cost {
                "custom_cost_json IS NOT NULL"
            } else {
                "custom_cost_json IS NULL"
            });
            has_where = true;
        }
        if let Some(custom_cost_applied) = request.custom_cost_applied {
            query.push(if has_where { " AND " } else { " WHERE " });
            query
                .push("custom_cost_applied = ")
                .push_bind(custom_cost_applied);
            has_where = true;
        }
        if let Some(metadata_key) = request.metadata_key.as_deref() {
            query.push(if has_where { " AND " } else { " WHERE " });
            query.push("metadata_json IS NOT NULL");
            query.push(" AND JSON_VALID(metadata_json)");
            query.push(" AND JSON_CONTAINS_PATH(metadata_json, 'one', CONCAT('$.', ");
            query.push_bind(metadata_key);
            query.push("))");
            if let Some(metadata_value) = request.metadata_value.as_deref() {
                query
                    .push(" AND JSON_UNQUOTE(JSON_EXTRACT(metadata_json, CONCAT('$.', ")
                    .push_bind(metadata_key)
                    .push("))) = ")
                    .push_bind(metadata_value);
            }
        }
        query
            .push(" ORDER BY timestamp_unix DESC LIMIT ")
            .push_bind(request.limit as i64);

        let rows = query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| RequestLogEntry {
                timestamp_unix: r.get::<i64, _>(0),
                api_key: r.get::<String, _>(1),
                project_id: r.get::<Option<String>, _>(2),
                session_id: r.get::<Option<String>, _>(3),
                metadata_json: r.get::<Option<String>, _>(4),
                custom_cost_json: r.get::<Option<String>, _>(5),
                custom_cost_applied: r.get::<bool, _>(6),
                provider_name: r.get::<Option<String>, _>(7),
                prompt_name: r.get::<Option<String>, _>(8),
                prompt_version: r.get::<Option<String>, _>(9),
                prompt_environment: r.get::<Option<String>, _>(10),
                model: r.get::<Option<String>, _>(11),
                input_tokens: r.get::<i64, _>(12) as u64,
                output_tokens: r.get::<i64, _>(13) as u64,
                cost: r.get::<f64, _>(14),
                is_streaming: r.get::<bool, _>(15),
                safety_mode: r.get::<Option<String>, _>(16),
                safety_matches: r.get::<Option<String>, _>(17),
                semantic_policy_version: r.get::<Option<String>, _>(18),
                semantic_index_state: r.get::<Option<String>, _>(19),
                semantic_degraded_reason: r.get::<Option<String>, _>(20),
                semantic_findings: r.get::<Option<String>, _>(21),
                tool_trace: r.get::<Option<String>, _>(22),
            })
            .collect())
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
        let mut builder = QueryBuilder::<MySql>::new(
            "SELECT session_id, project_id, project_ids_json, first_request_unix, last_request_unix, updated_at_unix, request_count, streaming_request_count, total_input_tokens, total_output_tokens, total_cost, providers_json, models_json, prompt_names_json, prompt_versions_json, tool_names_json, latest_request_json, safety_event_count, semantic_event_count, semantic_degraded_count, tool_call_count, tool_error_count, status, owner_id, owner_acquired_at_unix, last_transition_at_unix, last_transition_reason, last_heartbeat_unix, lease_expires_at_unix, cancel_requested_at_unix, cancel_requested_by, cancel_reason, handoff_target_owner_id, handoff_requested_at_unix, handoff_reason, state_json, metadata_json FROM session_state WHERE 1=1",
        );
        if let Some(project_id) = query.project_id.as_deref() {
            builder.push(" AND (project_id = ");
            builder.push_bind(project_id);
            builder.push(" OR project_ids_json LIKE ");
            builder.push_bind(format!("%\"{}\"%", project_id));
            builder.push(")");
        }
        if let Some(status) = query.status.as_deref() {
            builder.push(" AND LOWER(status) = LOWER(");
            builder.push_bind(status);
            builder.push(")");
        }
        if let Some(owner_id) = query.owner_id.as_deref() {
            builder.push(" AND owner_id = ");
            builder.push_bind(owner_id);
        }
        if let Some(updated_after_unix) = query.updated_after_unix {
            builder.push(" AND updated_at_unix >= ");
            builder.push_bind(updated_after_unix);
        }
        builder.push(" ORDER BY updated_at_unix DESC, COALESCE(last_request_unix, updated_at_unix) DESC LIMIT ");
        builder.push_bind(if query.limit == 0 {
            100_i64
        } else {
            query.limit as i64
        });

        let rows = builder
            .build()
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
             ON DUPLICATE KEY UPDATE \
               project_id = VALUES(project_id), \
               project_ids_json = VALUES(project_ids_json), \
               first_request_unix = VALUES(first_request_unix), \
               last_request_unix = VALUES(last_request_unix), \
               updated_at_unix = VALUES(updated_at_unix), \
               request_count = VALUES(request_count), \
               streaming_request_count = VALUES(streaming_request_count), \
               total_input_tokens = VALUES(total_input_tokens), \
               total_output_tokens = VALUES(total_output_tokens), \
               total_cost = VALUES(total_cost), \
               providers_json = VALUES(providers_json), \
               models_json = VALUES(models_json), \
               prompt_names_json = VALUES(prompt_names_json), \
               prompt_versions_json = VALUES(prompt_versions_json), \
               tool_names_json = VALUES(tool_names_json), \
               latest_request_json = VALUES(latest_request_json), \
               safety_event_count = VALUES(safety_event_count), \
               semantic_event_count = VALUES(semantic_event_count), \
               semantic_degraded_count = VALUES(semantic_degraded_count), \
               tool_call_count = VALUES(tool_call_count), \
               tool_error_count = VALUES(tool_error_count), \
               status = VALUES(status), \
               owner_id = VALUES(owner_id), \
               owner_acquired_at_unix = VALUES(owner_acquired_at_unix), \
               last_transition_at_unix = VALUES(last_transition_at_unix), \
               last_transition_reason = VALUES(last_transition_reason), \
               last_heartbeat_unix = VALUES(last_heartbeat_unix), \
               lease_expires_at_unix = VALUES(lease_expires_at_unix), \
               cancel_requested_at_unix = VALUES(cancel_requested_at_unix), \
               cancel_requested_by = VALUES(cancel_requested_by), \
               cancel_reason = VALUES(cancel_reason), \
               handoff_target_owner_id = VALUES(handoff_target_owner_id), \
               handoff_requested_at_unix = VALUES(handoff_requested_at_unix), \
               handoff_reason = VALUES(handoff_reason), \
               state_json = VALUES(state_json), \
               metadata_json = VALUES(metadata_json)",
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

        Ok(result.last_insert_id() as i64)
    }

    async fn get_session_events(
        &self,
        session_id: &str,
        after_seq: Option<i64>,
        limit: u32,
    ) -> Result<Vec<SessionEventRecord>, StoreError> {
        let mut builder = QueryBuilder::<MySql>::new(
            "SELECT event_seq, session_id, project_id, event_kind, actor_id, reason, payload_json, created_at_unix \
             FROM session_event WHERE session_id = ",
        );
        builder.push_bind(session_id);
        if let Some(after_seq) = after_seq {
            builder.push(" AND event_seq > ");
            builder.push_bind(after_seq);
        }
        builder.push(" ORDER BY event_seq ASC LIMIT ");
        builder.push_bind(if limit == 0 { 100_i64 } else { limit as i64 });

        let rows = builder
            .build()
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

    async fn get_virtual_key(
        &self,
        key_hash: &str,
    ) -> Result<Option<VirtualKeyRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT key_hash, project_id, name, provider_name, budget_limit, budget_duration,
                    budget_window_start, rpm_limit, tpm_limit, allowed_models, timeout_secs,
                    tool_approval_mode, allowed_tools, active, created_at, expires_at
             FROM virtual_keys
             WHERE key_hash = ?",
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
            active: r.get::<bool, _>(13),
            created_at: r.get::<String, _>(14),
            expires_at: r.get::<Option<String>, _>(15),
        }))
    }

    async fn get_all_virtual_keys(&self) -> Result<Vec<VirtualKeyRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT key_hash, project_id, name, provider_name, budget_limit, budget_duration,
                    budget_window_start, rpm_limit, tpm_limit, allowed_models, timeout_secs,
                    tool_approval_mode, allowed_tools, active, created_at, expires_at
             FROM virtual_keys",
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
                active: r.get::<bool, _>(13),
                created_at: r.get::<String, _>(14),
                expires_at: r.get::<Option<String>, _>(15),
            })
            .collect())
    }

    async fn upsert_virtual_key(&self, record: &VirtualKeyRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO virtual_keys
                 (key_hash, project_id, name, provider_name, budget_limit, budget_duration,
                  budget_window_start, rpm_limit, tpm_limit, allowed_models, timeout_secs,
                  tool_approval_mode, allowed_tools, active, created_at, expires_at)
             VALUES
                 (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 project_id = VALUES(project_id),
                 name = VALUES(name),
                 provider_name = VALUES(provider_name),
                 budget_limit = VALUES(budget_limit),
                 budget_duration = VALUES(budget_duration),
                 budget_window_start = VALUES(budget_window_start),
                 rpm_limit = VALUES(rpm_limit),
                 tpm_limit = VALUES(tpm_limit),
                 allowed_models = VALUES(allowed_models),
                 timeout_secs = VALUES(timeout_secs),
                 tool_approval_mode = VALUES(tool_approval_mode),
                 allowed_tools = VALUES(allowed_tools),
                 active = VALUES(active),
                 created_at = VALUES(created_at),
                 expires_at = VALUES(expires_at)",
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
        .bind(record.active)
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

    async fn get_project_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<ProjectPolicyRecord>, StoreError> {
        let row = sqlx::query(
            "SELECT project_id, budget_limit, budget_duration, rpm_limit, tpm_limit,
                    fallback_order, adaptive_enabled, timeout_secs, provider_rpm_limits,
                    provider_tpm_limits, provider_timeouts, provider_input_costs, provider_output_costs,
                    semantic_cache_enabled, semantic_cache_ttl_secs,
                    semantic_cache_similarity_threshold, tool_approval_mode, allowed_tools, updated_at
             FROM project_policies
             WHERE project_id = ?",
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
            adaptive_enabled: r.get::<bool, _>(6),
            timeout_secs: r.get::<Option<i64>, _>(7).map(|v| v as u64),
            provider_rpm_limits: r.get::<Option<String>, _>(8),
            provider_tpm_limits: r.get::<Option<String>, _>(9),
            provider_timeouts: r.get::<Option<String>, _>(10),
            provider_input_costs: r.get::<Option<String>, _>(11),
            provider_output_costs: r.get::<Option<String>, _>(12),
            semantic_cache_enabled: r.get::<Option<bool>, _>(13),
            semantic_cache_ttl_secs: r.get::<Option<i64>, _>(14).map(|v| v as u64),
            semantic_cache_similarity_threshold: r.get::<Option<f64>, _>(15),
            tool_approval_mode: r.get::<Option<String>, _>(16),
            allowed_tools: r.get::<Option<String>, _>(17),
            updated_at: r.get::<String, _>(18),
        }))
    }

    async fn get_all_project_policies(&self) -> Result<Vec<ProjectPolicyRecord>, StoreError> {
        let rows = sqlx::query(
            "SELECT project_id, budget_limit, budget_duration, rpm_limit, tpm_limit,
                    fallback_order, adaptive_enabled, timeout_secs, provider_rpm_limits,
                    provider_tpm_limits, provider_timeouts, provider_input_costs, provider_output_costs,
                    semantic_cache_enabled, semantic_cache_ttl_secs,
                    semantic_cache_similarity_threshold, tool_approval_mode, allowed_tools, updated_at
             FROM project_policies",
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
                adaptive_enabled: r.get::<bool, _>(6),
                timeout_secs: r.get::<Option<i64>, _>(7).map(|v| v as u64),
                provider_rpm_limits: r.get::<Option<String>, _>(8),
                provider_tpm_limits: r.get::<Option<String>, _>(9),
                provider_timeouts: r.get::<Option<String>, _>(10),
                provider_input_costs: r.get::<Option<String>, _>(11),
                provider_output_costs: r.get::<Option<String>, _>(12),
                semantic_cache_enabled: r.get::<Option<bool>, _>(13),
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
            "INSERT INTO project_policies
                 (project_id, budget_limit, budget_duration, rpm_limit, tpm_limit,
                  fallback_order, adaptive_enabled, timeout_secs, provider_rpm_limits,
                  provider_tpm_limits, provider_timeouts, provider_input_costs, provider_output_costs,
                  semantic_cache_enabled, semantic_cache_ttl_secs,
                  semantic_cache_similarity_threshold, tool_approval_mode, allowed_tools, updated_at)
             VALUES
                 (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(UNIX_TIMESTAMP() AS CHAR))
             ON DUPLICATE KEY UPDATE
                 budget_limit = VALUES(budget_limit),
                 budget_duration = VALUES(budget_duration),
                 rpm_limit = VALUES(rpm_limit),
                 tpm_limit = VALUES(tpm_limit),
                 fallback_order = VALUES(fallback_order),
                 adaptive_enabled = VALUES(adaptive_enabled),
                 timeout_secs = VALUES(timeout_secs),
                provider_rpm_limits = VALUES(provider_rpm_limits),
                provider_tpm_limits = VALUES(provider_tpm_limits),
                provider_timeouts = VALUES(provider_timeouts),
                provider_input_costs = VALUES(provider_input_costs),
                provider_output_costs = VALUES(provider_output_costs),
                semantic_cache_enabled = VALUES(semantic_cache_enabled),
                semantic_cache_ttl_secs = VALUES(semantic_cache_ttl_secs),
                semantic_cache_similarity_threshold = VALUES(semantic_cache_similarity_threshold),
                tool_approval_mode = VALUES(tool_approval_mode),
                allowed_tools = VALUES(allowed_tools),
                updated_at = VALUES(updated_at)",
        )
        .bind(&record.project_id)
        .bind(record.budget_limit)
        .bind(&record.budget_duration)
        .bind(record.rpm_limit.map(|v| v as i32))
        .bind(record.tpm_limit.map(|v| v as i32))
        .bind(&record.fallback_order)
        .bind(record.adaptive_enabled)
        .bind(record.timeout_secs.map(|v| v as i64))
        .bind(&record.provider_rpm_limits)
        .bind(&record.provider_tpm_limits)
        .bind(&record.provider_timeouts)
        .bind(&record.provider_input_costs)
        .bind(&record.provider_output_costs)
        .bind(record.semantic_cache_enabled)
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
            enabled: r.get::<bool, _>(1),
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
                enabled: r.get::<bool, _>(1),
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
            "INSERT INTO managed_providers
                 (name, enabled, api_key_env, base_url, models_json, api_key_header, timeout_secs, family, surfaces_json, routing_metadata_json, created_at, updated_at)
             VALUES
                 (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 enabled = VALUES(enabled),
                 api_key_env = VALUES(api_key_env),
                 base_url = VALUES(base_url),
                 models_json = VALUES(models_json),
                 api_key_header = VALUES(api_key_header),
                 timeout_secs = VALUES(timeout_secs),
                 family = VALUES(family),
                 surfaces_json = VALUES(surfaces_json),
                 routing_metadata_json = VALUES(routing_metadata_json),
                 created_at = VALUES(created_at),
                 updated_at = VALUES(updated_at)",
        )
        .bind(&record.name)
        .bind(record.enabled)
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
        let mut query = QueryBuilder::<MySql>::new(
            "SELECT rule_id, project_id, name, priority, enabled, match_path, match_model,
                    match_streaming, match_role, match_headers, min_prompt_tokens,
                    max_prompt_tokens, deny_reason, provider_order, provider_weights,
                    timeout_secs, created_at
             FROM routing_rules",
        );
        if let Some(project_id) = project_id {
            query.push(" WHERE project_id = ").push_bind(project_id);
        }
        query.push(" ORDER BY priority DESC, created_at ASC");

        let rows = query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StoreError::Db(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| RoutingRuleRecord {
                rule_id: r.get::<String, _>(0),
                project_id: r.get::<String, _>(1),
                name: r.get::<String, _>(2),
                priority: r.get::<i32, _>(3),
                enabled: r.get::<bool, _>(4),
                match_path: r.get::<Option<String>, _>(5),
                match_model: r.get::<Option<String>, _>(6),
                match_streaming: r.get::<Option<bool>, _>(7),
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
            "INSERT INTO routing_rules
                 (rule_id, project_id, name, priority, enabled, match_path, match_model,
                  match_streaming, match_role, match_headers, min_prompt_tokens,
                  max_prompt_tokens, deny_reason, provider_order, provider_weights,
                  timeout_secs, created_at)
             VALUES
                 (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 project_id = VALUES(project_id),
                 name = VALUES(name),
                 priority = VALUES(priority),
                 enabled = VALUES(enabled),
                 match_path = VALUES(match_path),
                 match_model = VALUES(match_model),
                 match_streaming = VALUES(match_streaming),
                 match_role = VALUES(match_role),
                 match_headers = VALUES(match_headers),
                 min_prompt_tokens = VALUES(min_prompt_tokens),
                 max_prompt_tokens = VALUES(max_prompt_tokens),
                 deny_reason = VALUES(deny_reason),
                 provider_order = VALUES(provider_order),
                 provider_weights = VALUES(provider_weights),
                 timeout_secs = VALUES(timeout_secs),
                 created_at = VALUES(created_at)",
        )
        .bind(&record.rule_id)
        .bind(&record.project_id)
        .bind(&record.name)
        .bind(record.priority)
        .bind(record.enabled)
        .bind(&record.match_path)
        .bind(&record.match_model)
        .bind(record.match_streaming)
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
            "SELECT project_id, mode, rules_json, updated_at
             FROM safety_policies
             WHERE project_id = ?",
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
            "INSERT INTO safety_policies (project_id, mode, rules_json, updated_at)
             VALUES (?, ?, ?, CAST(UNIX_TIMESTAMP() AS CHAR))
             ON DUPLICATE KEY UPDATE
                 mode = VALUES(mode),
                 rules_json = VALUES(rules_json),
                 updated_at = VALUES(updated_at)",
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
            enabled: r.get::<bool, _>(2),
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
                enabled: r.get::<bool, _>(2),
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
            "INSERT INTO semantic_policies (project_id, version, enabled, entities_json, topics_json, updated_at) VALUES (?, ?, ?, ?, ?, ?) \
             ON DUPLICATE KEY UPDATE version = VALUES(version), enabled = VALUES(enabled), entities_json = VALUES(entities_json), topics_json = VALUES(topics_json), updated_at = VALUES(updated_at)",
        )
        .bind(&record.project_id)
        .bind(&record.version)
        .bind(record.enabled)
        .bind(&record.entities_json)
        .bind(&record.topics_json)
        .bind(&record.updated_at)
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
            "INSERT INTO prompt_cache_routes
                (route_id, project_id, cache_key, provider_name, signal_kind, signal_strength,
                 observed_at_ms, expires_at_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 project_id = VALUES(project_id),
                 cache_key = VALUES(cache_key),
                 provider_name = VALUES(provider_name),
                 signal_kind = VALUES(signal_kind),
                 signal_strength = VALUES(signal_strength),
                 observed_at_ms = VALUES(observed_at_ms),
                 expires_at_ms = VALUES(expires_at_ms)",
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
                response_status: r.get::<i32, _>(5) as u16,
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
            "INSERT INTO semantic_cache_entries
                (cache_id, project_id, provider_name, model, tokens_json, response_status,
                 content_type, response_body, prompt_tokens, created_at_ms, expires_at_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 project_id = VALUES(project_id),
                 provider_name = VALUES(provider_name),
                 model = VALUES(model),
                 tokens_json = VALUES(tokens_json),
                 response_status = VALUES(response_status),
                 content_type = VALUES(content_type),
                 response_body = VALUES(response_body),
                 prompt_tokens = VALUES(prompt_tokens),
                 created_at_ms = VALUES(created_at_ms),
                 expires_at_ms = VALUES(expires_at_ms)",
        )
        .bind(&record.cache_id)
        .bind(&record.project_id)
        .bind(&record.provider_name)
        .bind(&record.model)
        .bind(&record.tokens_json)
        .bind(record.response_status as i32)
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
                 FROM (
                     SELECT cache_id
                     FROM semantic_cache_entries
                     ORDER BY created_at_ms ASC
                     LIMIT ?
                 ) AS doomed
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
            enabled: r.get::<bool, _>(6),
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
                enabled: r.get::<bool, _>(6),
                timeout_ms: r.get::<Option<i64>, _>(7).map(|value| value as u64),
                updated_at: r.get::<String, _>(8),
            })
            .collect())
    }

    async fn upsert_project_tool(&self, record: &ProjectToolRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO project_tools (project_id, tool_name, description, input_schema_json, executor_kind, executor_config_json, enabled, timeout_ms, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 description = VALUES(description),
                 input_schema_json = VALUES(input_schema_json),
                 executor_kind = VALUES(executor_kind),
                 executor_config_json = VALUES(executor_config_json),
                 enabled = VALUES(enabled),
                 timeout_ms = VALUES(timeout_ms),
                 updated_at = VALUES(updated_at)",
        )
        .bind(&record.project_id)
        .bind(&record.tool_name)
        .bind(&record.description)
        .bind(&record.input_schema_json)
        .bind(&record.executor_kind)
        .bind(&record.executor_config_json)
        .bind(record.enabled)
        .bind(record.timeout_ms.map(|value| value as i64))
        .bind(&record.updated_at)
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
            active: r.get::<bool, _>(9),
            updated_at: r.get::<String, _>(10),
        }))
    }

    async fn get_project_prompts(
        &self,
        project_id: Option<&str>,
        prompt_name: Option<&str>,
    ) -> Result<Vec<ProjectPromptRecord>, StoreError> {
        let mut query = QueryBuilder::<MySql>::new(
            "SELECT project_id, prompt_name, version, environment, description, target, template_text, variables_schema_json, rollout_metadata_json, active, updated_at
             FROM project_prompts",
        );

        let mut has_where = false;
        if let Some(project_id) = project_id {
            query.push(" WHERE project_id = ").push_bind(project_id);
            has_where = true;
        }
        if let Some(prompt_name) = prompt_name {
            query.push(if has_where { " AND " } else { " WHERE " });
            query.push("prompt_name = ").push_bind(prompt_name);
        }
        query.push(" ORDER BY project_id, prompt_name, environment, version");

        let rows = query
            .build()
            .fetch_all(&self.pool)
            .await
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
                active: r.get::<bool, _>(9),
                updated_at: r.get::<String, _>(10),
            })
            .collect())
    }

    async fn upsert_project_prompt(&self, record: &ProjectPromptRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO project_prompts (project_id, prompt_name, version, environment, description, target, template_text, variables_schema_json, rollout_metadata_json, active, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 environment = VALUES(environment),
                 description = VALUES(description),
                 target = VALUES(target),
                 template_text = VALUES(template_text),
                 variables_schema_json = VALUES(variables_schema_json),
                 rollout_metadata_json = VALUES(rollout_metadata_json),
                 active = VALUES(active),
                 updated_at = VALUES(updated_at)",
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
        .bind(record.active)
        .bind(&record.updated_at)
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
            "INSERT INTO project_rollout_policies (project_id, policy_name, description, gate_config_json, target_environment, updated_at)
             VALUES (?, ?, ?, ?, ?, CAST(UNIX_TIMESTAMP() AS CHAR))
             ON DUPLICATE KEY UPDATE
                 description = VALUES(description),
                 gate_config_json = VALUES(gate_config_json),
                 target_environment = VALUES(target_environment),
                 updated_at = VALUES(updated_at)",
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
            "INSERT INTO project_prompt_rollouts (project_id, prompt_name, rollout_id, policy_name, baseline_version, candidate_version, baseline_run_id, candidate_run_id, target_environment, status, recommendation_action, comparison_json, created_at, applied_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 policy_name = VALUES(policy_name),
                 baseline_version = VALUES(baseline_version),
                 candidate_version = VALUES(candidate_version),
                 baseline_run_id = VALUES(baseline_run_id),
                 candidate_run_id = VALUES(candidate_run_id),
                 target_environment = VALUES(target_environment),
                 status = VALUES(status),
                 recommendation_action = VALUES(recommendation_action),
                 comparison_json = VALUES(comparison_json),
                 created_at = VALUES(created_at),
                 applied_at = VALUES(applied_at)",
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
            "INSERT INTO project_datasets (project_id, dataset_name, description, schema_json, updated_at)
             VALUES (?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 description = VALUES(description),
                 schema_json = VALUES(schema_json),
                 updated_at = VALUES(updated_at)",
        )
        .bind(&record.project_id)
        .bind(&record.dataset_name)
        .bind(&record.description)
        .bind(&record.schema_json)
        .bind(&record.updated_at)
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
            "INSERT INTO project_dataset_items (project_id, dataset_name, item_id, input_json, expected_output_json, metadata_json, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 input_json = VALUES(input_json),
                 expected_output_json = VALUES(expected_output_json),
                 metadata_json = VALUES(metadata_json),
                 updated_at = VALUES(updated_at)",
        )
        .bind(&record.project_id)
        .bind(&record.dataset_name)
        .bind(&record.item_id)
        .bind(&record.input_json)
        .bind(&record.expected_output_json)
        .bind(&record.metadata_json)
        .bind(&record.updated_at)
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
            total_items: r.get::<i32, _>(5) as u32,
            passed_items: r.get::<i32, _>(6) as u32,
            failed_items: r.get::<i32, _>(7) as u32,
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
                total_items: r.get::<i32, _>(5) as u32,
                passed_items: r.get::<i32, _>(6) as u32,
                failed_items: r.get::<i32, _>(7) as u32,
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
        let mut query = sqlx::QueryBuilder::<sqlx::MySql>::new(
            "SELECT run_id, project_id, dataset_name, target_url, status, total_items, passed_items, failed_items,
                    total_input_tokens, total_output_tokens, total_cost, average_latency_ms, summary_json, created_at, completed_at
             FROM project_eval_runs
             WHERE status IN (",
        );
        {
            let mut separated = query.separated(", ");
            for status in statuses {
                separated.push_bind(status);
            }
        }
        query.push(") ORDER BY created_at ASC, run_id ASC LIMIT ");
        query.push_bind(limit);
        let rows = query
            .build()
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
                total_items: r.get::<i32, _>(5) as u32,
                passed_items: r.get::<i32, _>(6) as u32,
                failed_items: r.get::<i32, _>(7) as u32,
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
            "INSERT INTO project_eval_runs
             (run_id, project_id, dataset_name, target_url, status, total_items, passed_items, failed_items,
              total_input_tokens, total_output_tokens, total_cost, average_latency_ms, summary_json, created_at, completed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 project_id = VALUES(project_id),
                 dataset_name = VALUES(dataset_name),
                 target_url = VALUES(target_url),
                 status = VALUES(status),
                 total_items = VALUES(total_items),
                 passed_items = VALUES(passed_items),
                 failed_items = VALUES(failed_items),
                 total_input_tokens = VALUES(total_input_tokens),
                 total_output_tokens = VALUES(total_output_tokens),
                 total_cost = VALUES(total_cost),
                 average_latency_ms = VALUES(average_latency_ms),
                 summary_json = VALUES(summary_json),
                 created_at = VALUES(created_at),
                 completed_at = VALUES(completed_at)",
        )
        .bind(&record.run_id)
        .bind(&record.project_id)
        .bind(&record.dataset_name)
        .bind(&record.target_url)
        .bind(&record.status)
        .bind(record.total_items as i32)
        .bind(record.passed_items as i32)
        .bind(record.failed_items as i32)
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
                passed: r.get::<bool, _>(4),
                status_code: r.get::<Option<i32>, _>(5).map(|value| value as u16),
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
            "INSERT INTO project_eval_run_items
             (run_id, project_id, dataset_name, item_id, passed, status_code, latency_ms, output_text,
              evaluation_json, error, input_tokens, output_tokens, cost, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 project_id = VALUES(project_id),
                 dataset_name = VALUES(dataset_name),
                 passed = VALUES(passed),
                 status_code = VALUES(status_code),
                 latency_ms = VALUES(latency_ms),
                 output_text = VALUES(output_text),
                 evaluation_json = VALUES(evaluation_json),
                 error = VALUES(error),
                 input_tokens = VALUES(input_tokens),
                 output_tokens = VALUES(output_tokens),
                 cost = VALUES(cost),
                 created_at = VALUES(created_at)",
        )
        .bind(&record.run_id)
        .bind(&record.project_id)
        .bind(&record.dataset_name)
        .bind(&record.item_id)
        .bind(record.passed)
        .bind(record.status_code.map(i32::from))
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
            "INSERT INTO governance_history
             (change_id, project_id, resource_type, resource_id, action, before_json, after_json, changed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE
                 project_id = VALUES(project_id),
                 resource_type = VALUES(resource_type),
                 resource_id = VALUES(resource_id),
                 action = VALUES(action),
                 before_json = VALUES(before_json),
                 after_json = VALUES(after_json),
                 changed_at = VALUES(changed_at)",
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
        let mut query = QueryBuilder::<MySql>::new(
            "SELECT change_id, project_id, resource_type, resource_id, action, before_json, after_json, changed_at
             FROM governance_history
             WHERE project_id = ",
        );
        query.push_bind(project_id);
        if let Some(resource_type) = resource_type {
            query.push(" AND resource_type = ").push_bind(resource_type);
        }
        query
            .push(" ORDER BY changed_at DESC, change_id DESC LIMIT ")
            .push_bind(limit as i64);
        let rows = query
            .build()
            .fetch_all(&self.pool)
            .await
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
