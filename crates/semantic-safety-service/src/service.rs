use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use semantic_safety_protocol::{
    Chunk, DeleteProjectPolicyRequest, DeleteProjectPolicyResponse, EvaluateRequest,
    EvaluateResponse, Finding, GetProjectSyncStatusRequest, GetProjectSyncStatusResponse,
    HealthRequest, HealthResponse, IndexState, ListProjectSyncStatesRequest,
    ListProjectSyncStatesResponse, ProjectSemanticPolicy, ProjectSyncState, SemanticEntity,
    SemanticSafetyService, UpsertProjectPolicyRequest, UpsertProjectPolicyResponse,
};
use tonic::{Request, Response, Status};

use crate::backend::InferenceBackend;
use crate::metrics::SemanticSafetyServiceMetrics;
use crate::persistence::{FileProjectIndexStore, PersistedProjectIndex};

#[derive(Clone)]
pub struct SemanticSafetyConfig {
    pub auth_token: Option<String>,
}

#[derive(Clone)]
pub struct CompiledEntity {
    pub entity: SemanticEntity,
    pub aliases: Vec<String>,
}

#[derive(Clone)]
pub struct TopicExemplar {
    pub topic_id: String,
    pub exemplar: String,
    pub embedding: Vec<f32>,
    pub rerank_threshold: f32,
    pub require_entity_match: bool,
}

#[derive(Clone)]
pub struct CompiledProjectIndex {
    pub policy: ProjectSemanticPolicy,
    pub entities: Vec<CompiledEntity>,
    pub exemplars: Vec<TopicExemplar>,
    pub stored_exemplar_count: u64,
}

#[derive(Clone)]
pub struct SemanticSafetyGrpcService {
    config: SemanticSafetyConfig,
    store: Arc<FileProjectIndexStore>,
    backend: Arc<dyn InferenceBackend>,
    indexes: Arc<DashMap<String, CompiledProjectIndex>>,
    metrics: Option<Arc<SemanticSafetyServiceMetrics>>,
}

impl SemanticSafetyGrpcService {
    pub fn new(
        config: SemanticSafetyConfig,
        store: Arc<FileProjectIndexStore>,
        backend: Arc<dyn InferenceBackend>,
    ) -> Self {
        Self {
            config,
            store,
            backend,
            indexes: Arc::new(DashMap::new()),
            metrics: None,
        }
    }

    pub fn with_metrics(mut self, metrics: Arc<SemanticSafetyServiceMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub async fn load_from_store(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut compiled_indexes = Vec::new();
        for record in self.store.load_all()? {
            let compiled = self
                .compile_policy(record.policy, Some(record.exemplar_embeddings))
                .await?;
            compiled_indexes.push(compiled);
        }

        self.indexes.clear();
        for compiled in compiled_indexes {
            self.indexes
                .insert(compiled.policy.project_id.clone(), compiled);
        }
        self.refresh_index_metrics();
        Ok(())
    }

    async fn compile_policy(
        &self,
        policy: ProjectSemanticPolicy,
        exemplar_embeddings: Option<Vec<Vec<f32>>>,
    ) -> Result<CompiledProjectIndex, Box<dyn std::error::Error + Send + Sync>> {
        let mut exemplar_texts = Vec::new();
        let mut exemplar_meta = Vec::new();
        for topic in &policy.topics {
            for exemplar in &topic.exemplars {
                exemplar_texts.push(normalize_text(exemplar));
                exemplar_meta.push((
                    topic.topic_id.clone(),
                    exemplar.clone(),
                    topic.rerank_threshold,
                    topic.require_entity_match,
                ));
            }
        }

        let embeddings = match exemplar_embeddings {
            Some(existing) if existing.len() == exemplar_texts.len() => existing,
            _ => self.backend.embed_texts(&exemplar_texts).await?,
        };

        let exemplars = exemplar_meta
            .into_iter()
            .zip(embeddings.into_iter())
            .map(
                |((topic_id, exemplar, rerank_threshold, require_entity_match), embedding)| {
                    TopicExemplar {
                        topic_id,
                        exemplar,
                        embedding,
                        rerank_threshold,
                        require_entity_match,
                    }
                },
            )
            .collect::<Vec<_>>();

        let entities = policy
            .entities
            .iter()
            .cloned()
            .map(|entity| {
                let mut aliases = vec![normalize_text(&entity.name)];
                aliases.extend(entity.aliases.iter().map(|alias| normalize_text(alias)));
                aliases.sort();
                aliases.dedup();
                CompiledEntity { entity, aliases }
            })
            .collect::<Vec<_>>();

        Ok(CompiledProjectIndex {
            stored_exemplar_count: exemplars.len() as u64,
            policy,
            entities,
            exemplars,
        })
    }

    #[expect(
        clippy::result_large_err,
        reason = "gRPC authorization failures are returned as tonic::Status"
    )]
    fn authorize<T>(&self, request: &Request<T>) -> Result<(), Status> {
        if let Some(expected) = &self.config.auth_token {
            let actual = request
                .metadata()
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default();
            if actual != format!("Bearer {expected}") {
                return Err(Status::unauthenticated("invalid semantic safety token"));
            }
        }
        Ok(())
    }

    fn find_entity_matches(index: &CompiledProjectIndex, text: &str) -> Vec<(String, String)> {
        let normalized = normalize_text(text);
        index
            .entities
            .iter()
            .filter_map(|entity| {
                entity.aliases.iter().find_map(|alias| {
                    if contains_exact_alias(&normalized, alias) {
                        Some((entity.entity.entity_id.clone(), alias.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        if left.len() != right.len() || left.is_empty() {
            return 0.0;
        }
        let dot = left
            .iter()
            .zip(right.iter())
            .map(|(l, r)| l * r)
            .sum::<f32>();
        let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
        let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
        if left_norm == 0.0 || right_norm == 0.0 {
            0.0
        } else {
            dot / (left_norm * right_norm)
        }
    }

    async fn evaluate_chunk(
        &self,
        index: &CompiledProjectIndex,
        chunk: &Chunk,
        chunk_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<Finding>, Box<dyn std::error::Error + Send + Sync>> {
        let entity_matches = Self::find_entity_matches(index, &chunk.text);

        let mut ranked = index
            .exemplars
            .iter()
            .map(|exemplar| {
                (
                    exemplar,
                    Self::cosine_similarity(chunk_embedding, &exemplar.embedding),
                )
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
        ranked.truncate(top_k.max(1));

        let rerank_inputs = ranked
            .iter()
            .map(|(exemplar, _)| exemplar.exemplar.clone())
            .collect::<Vec<_>>();
        let rerank_scores = self.backend.rerank(&chunk.text, &rerank_inputs).await?;

        let mut findings = Vec::new();
        for (((exemplar, embedding_score), rerank_input), rerank_score) in ranked
            .into_iter()
            .zip(rerank_inputs.into_iter())
            .zip(rerank_scores.into_iter())
        {
            if exemplar.require_entity_match && entity_matches.is_empty() {
                continue;
            }
            if rerank_score < exemplar.rerank_threshold {
                continue;
            }
            let (entity_id, entity_text) = entity_matches
                .first()
                .cloned()
                .unwrap_or_else(|| ("".to_string(), "".to_string()));
            findings.push(Finding {
                chunk_path: chunk.path.clone(),
                entity_id,
                entity_text,
                topic_id: exemplar.topic_id.clone(),
                matched_exemplar: rerank_input,
                embedding_score,
                rerank_score,
            });
        }
        Ok(findings)
    }

    fn build_evaluate_response(
        started: Instant,
        project_id: String,
        policy_version: String,
        findings: Vec<Finding>,
        index_state: IndexState,
        degraded_reason: impl Into<String>,
    ) -> EvaluateResponse {
        EvaluateResponse {
            project_id,
            policy_version,
            findings,
            service_latency_ms: started.elapsed().as_millis() as u64,
            index_state: index_state as i32,
            degraded_reason: degraded_reason.into(),
        }
    }

    fn refresh_index_metrics(&self) {
        let Some(metrics) = &self.metrics else {
            return;
        };
        let mut projects = 0i64;
        let mut exemplars = 0i64;
        for entry in self.indexes.iter() {
            projects += 1;
            exemplars += entry.stored_exemplar_count as i64;
        }
        metrics.indexed_projects.set(projects);
        metrics.indexed_exemplars.set(exemplars);
    }

    fn observe_evaluate(&self, outcome: &str, started: Instant) {
        let Some(metrics) = &self.metrics else {
            return;
        };
        metrics
            .evaluate_requests_total
            .with_label_values(&[outcome])
            .inc();
        metrics
            .evaluate_latency_ms
            .with_label_values(&[outcome])
            .observe(started.elapsed().as_millis() as f64);
    }
}

#[tonic::async_trait]
impl SemanticSafetyService for SemanticSafetyGrpcService {
    async fn evaluate(
        &self,
        request: Request<EvaluateRequest>,
    ) -> Result<Response<EvaluateResponse>, Status> {
        self.authorize(&request)?;
        let started = Instant::now();
        let payload = request.into_inner();
        let Some(index) = self.indexes.get(&payload.project_id) else {
            self.observe_evaluate("missing", started);
            return Ok(Response::new(Self::build_evaluate_response(
                started,
                payload.project_id,
                payload.policy_version,
                Vec::new(),
                IndexState::Missing,
                String::new(),
            )));
        };

        if !index.policy.enabled {
            self.observe_evaluate("disabled", started);
            return Ok(Response::new(Self::build_evaluate_response(
                started,
                index.policy.project_id.clone(),
                index.policy.version.clone(),
                Vec::new(),
                IndexState::Disabled,
                String::new(),
            )));
        }

        let chunk_texts = payload
            .chunks
            .iter()
            .map(|chunk| normalize_text(&chunk.text))
            .collect::<Vec<_>>();
        let backend_health = self.backend.health();
        let embeddings = match self.backend.embed_texts(&chunk_texts).await {
            Ok(embeddings) => embeddings,
            Err(error) => {
                self.observe_evaluate("degraded", started);
                return Ok(Response::new(Self::build_evaluate_response(
                    started,
                    index.policy.project_id.clone(),
                    index.policy.version.clone(),
                    Vec::new(),
                    IndexState::Degraded,
                    format!("embedding failed: {error}"),
                )));
            }
        };
        if embeddings.len() != payload.chunks.len() {
            self.observe_evaluate("degraded", started);
            return Ok(Response::new(Self::build_evaluate_response(
                started,
                index.policy.project_id.clone(),
                index.policy.version.clone(),
                Vec::new(),
                IndexState::Degraded,
                format!(
                    "embedding count mismatch: expected {} embeddings for {} chunks, got {}",
                    payload.chunks.len(),
                    payload.chunks.len(),
                    embeddings.len()
                ),
            )));
        }

        let mut findings = Vec::new();
        for (chunk, embedding) in payload.chunks.iter().zip(embeddings.iter()) {
            match self
                .evaluate_chunk(&index, chunk, embedding, payload.top_k as usize)
                .await
            {
                Ok(chunk_findings) => findings.extend(chunk_findings),
                Err(error) => {
                    self.observe_evaluate("degraded", started);
                    return Ok(Response::new(Self::build_evaluate_response(
                        started,
                        index.policy.project_id.clone(),
                        index.policy.version.clone(),
                        Vec::new(),
                        IndexState::Degraded,
                        format!("rerank failed: {error}"),
                    )));
                }
            }
        }

        let base_index_state = if index.policy.version == payload.policy_version {
            IndexState::Ready
        } else {
            IndexState::Stale
        };
        let (index_state, degraded_reason) =
            if matches!(base_index_state, IndexState::Ready) && !backend_health.ready {
                (IndexState::Degraded, backend_health.message)
            } else {
                (base_index_state, String::new())
            };

        let outcome = match index_state {
            IndexState::Ready => "ready",
            IndexState::Stale => "stale",
            IndexState::Disabled => "disabled",
            IndexState::Missing => "missing",
            IndexState::Degraded => "degraded",
            IndexState::Unspecified => "unspecified",
        };
        self.observe_evaluate(outcome, started);

        Ok(Response::new(Self::build_evaluate_response(
            started,
            index.policy.project_id.clone(),
            index.policy.version.clone(),
            findings,
            index_state,
            degraded_reason,
        )))
    }

    async fn upsert_project_policy(
        &self,
        request: Request<UpsertProjectPolicyRequest>,
    ) -> Result<Response<UpsertProjectPolicyResponse>, Status> {
        self.authorize(&request)?;
        let payload = request.into_inner();
        let policy = payload
            .policy
            .ok_or_else(|| Status::invalid_argument("policy is required"))?;
        let compiled = self
            .compile_policy(policy.clone(), None)
            .await
            .map_err(|error| Status::internal(error.to_string()))?;
        self.store
            .upsert(PersistedProjectIndex {
                policy: policy.clone(),
                exemplar_embeddings: compiled
                    .exemplars
                    .iter()
                    .map(|exemplar| exemplar.embedding.clone())
                    .collect(),
                stored_exemplar_count: compiled.stored_exemplar_count,
            })
            .map_err(|error| Status::internal(error.to_string()))?;
        self.indexes.insert(policy.project_id.clone(), compiled);
        self.refresh_index_metrics();
        Ok(Response::new(UpsertProjectPolicyResponse {
            ok: true,
            project_id: policy.project_id,
            policy_version: policy.version,
        }))
    }

    async fn delete_project_policy(
        &self,
        request: Request<DeleteProjectPolicyRequest>,
    ) -> Result<Response<DeleteProjectPolicyResponse>, Status> {
        self.authorize(&request)?;
        let payload = request.into_inner();
        self.store
            .delete(&payload.project_id)
            .map_err(|error| Status::internal(error.to_string()))?;
        self.indexes.remove(&payload.project_id);
        self.refresh_index_metrics();
        Ok(Response::new(DeleteProjectPolicyResponse { ok: true }))
    }

    async fn get_project_sync_status(
        &self,
        request: Request<GetProjectSyncStatusRequest>,
    ) -> Result<Response<GetProjectSyncStatusResponse>, Status> {
        self.authorize(&request)?;
        let payload = request.into_inner();
        if let Some(index) = self.indexes.get(&payload.project_id) {
            return Ok(Response::new(GetProjectSyncStatusResponse {
                project_id: index.policy.project_id.clone(),
                policy_version: index.policy.version.clone(),
                index_state: if index.policy.enabled {
                    IndexState::Ready as i32
                } else {
                    IndexState::Disabled as i32
                },
                updated_at: index.policy.updated_at.clone(),
                stored_exemplar_count: index.stored_exemplar_count,
            }));
        }
        Ok(Response::new(GetProjectSyncStatusResponse {
            project_id: payload.project_id,
            policy_version: String::new(),
            index_state: IndexState::Missing as i32,
            updated_at: String::new(),
            stored_exemplar_count: 0,
        }))
    }

    async fn list_project_sync_states(
        &self,
        request: Request<ListProjectSyncStatesRequest>,
    ) -> Result<Response<ListProjectSyncStatesResponse>, Status> {
        self.authorize(&request)?;

        let mut projects = self
            .indexes
            .iter()
            .map(|entry| ProjectSyncState {
                project_id: entry.policy.project_id.clone(),
                policy_version: entry.policy.version.clone(),
                index_state: if entry.policy.enabled {
                    IndexState::Ready as i32
                } else {
                    IndexState::Disabled as i32
                },
                updated_at: entry.policy.updated_at.clone(),
                stored_exemplar_count: entry.stored_exemplar_count,
            })
            .collect::<Vec<_>>();
        projects.sort_by(|left, right| left.project_id.cmp(&right.project_id));

        Ok(Response::new(ListProjectSyncStatesResponse { projects }))
    }

    async fn health(
        &self,
        request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        self.authorize(&request)?;
        let health = self.backend.health();
        Ok(Response::new(HealthResponse {
            ready: health.ready,
            backend: health.backend.to_string(),
            message: health.message,
        }))
    }
}

fn normalize_text(input: &str) -> String {
    input
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn contains_exact_alias(text: &str, alias: &str) -> bool {
    if alias.is_empty() {
        return false;
    }

    let mut search_from = 0;
    while let Some(offset) = text[search_from..].find(alias) {
        let start = search_from + offset;
        let end = start + alias.len();
        let starts_at_boundary = text[..start]
            .chars()
            .next_back()
            .map(|ch| !ch.is_ascii_alphanumeric())
            .unwrap_or(true);
        let ends_at_boundary = text[end..]
            .chars()
            .next()
            .map(|ch| !ch.is_ascii_alphanumeric())
            .unwrap_or(true);
        if starts_at_boundary && ends_at_boundary {
            return true;
        }
        search_from = start
            + text[start..]
                .chars()
                .next()
                .map(|ch| ch.len_utf8())
                .unwrap_or(1);
    }

    false
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use semantic_safety_protocol::{
        Chunk, EvaluateRequest, HealthRequest, IndexState, ProjectSemanticPolicy, SemanticEntity,
        SemanticTopic,
    };
    use tempfile::tempdir;
    use tonic::Request;

    use super::*;
    use crate::backend::{BackendHealth, InferenceBackend, TensorRtBackend};
    use crate::metrics::SemanticSafetyServiceMetrics;

    struct FailingBackend;

    #[derive(Default)]
    struct CountingBackend {
        embed_calls: AtomicUsize,
        rerank_calls: AtomicUsize,
    }

    #[derive(Default)]
    struct ShortEmbeddingBackend;

    #[async_trait]
    impl InferenceBackend for FailingBackend {
        async fn embed_texts(
            &self,
            texts: &[String],
        ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }

        async fn rerank(
            &self,
            _query: &str,
            _candidates: &[String],
        ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            Err("reranker exploded".into())
        }

        fn backend_name(&self) -> &'static str {
            "failing"
        }

        fn health(&self) -> BackendHealth {
            BackendHealth {
                ready: true,
                backend: "failing",
                message: "runner configured".to_string(),
            }
        }
    }

    #[async_trait]
    impl InferenceBackend for CountingBackend {
        async fn embed_texts(
            &self,
            texts: &[String],
        ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
            self.embed_calls.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }

        async fn rerank(
            &self,
            _query: &str,
            candidates: &[String],
        ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            self.rerank_calls.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(candidates.iter().map(|_| 0.95).collect())
        }

        fn backend_name(&self) -> &'static str {
            "counting"
        }

        fn health(&self) -> BackendHealth {
            BackendHealth {
                ready: true,
                backend: "counting",
                message: "counting backend ready".to_string(),
            }
        }
    }

    #[async_trait]
    impl InferenceBackend for ShortEmbeddingBackend {
        async fn embed_texts(
            &self,
            _texts: &[String],
        ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![vec![1.0, 0.0]])
        }

        async fn rerank(
            &self,
            _query: &str,
            _candidates: &[String],
        ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![0.5])
        }

        fn backend_name(&self) -> &'static str {
            "short"
        }

        fn health(&self) -> BackendHealth {
            BackendHealth {
                ready: true,
                backend: "short",
                message: "short backend ready".to_string(),
            }
        }
    }

    fn sample_policy() -> ProjectSemanticPolicy {
        ProjectSemanticPolicy {
            project_id: "project-a".to_string(),
            version: "1".to_string(),
            enabled: true,
            entities: vec![SemanticEntity {
                entity_id: "company-x".to_string(),
                name: "Company X".to_string(),
                aliases: vec!["companyx".to_string()],
            }],
            topics: vec![SemanticTopic {
                topic_id: "layoffs".to_string(),
                name: "Layoffs".to_string(),
                exemplars: vec!["company x layoffs next week".to_string()],
                rerank_threshold: 0.1,
                require_entity_match: true,
            }],
            updated_at: "1".to_string(),
        }
    }

    #[tokio::test]
    async fn entity_required_topics_only_fire_with_entity_match() {
        let dir = tempdir().unwrap();
        let service = SemanticSafetyGrpcService::new(
            SemanticSafetyConfig { auth_token: None },
            Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap()),
            Arc::new(TensorRtBackend::new_dev_stub()),
        );
        service
            .upsert_project_policy(Request::new(UpsertProjectPolicyRequest {
                policy: Some(sample_policy()),
            }))
            .await
            .unwrap();

        let response = service
            .evaluate(Request::new(EvaluateRequest {
                project_id: "project-a".to_string(),
                policy_version: "1".to_string(),
                request_id: "req-1".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4o".to_string(),
                streaming: false,
                chunks: vec![Chunk {
                    path: "$.messages[0].content".to_string(),
                    text: "Layoffs next week".to_string(),
                }],
                top_k: 3,
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(response.findings.is_empty());
        assert_eq!(response.index_state, IndexState::Degraded as i32);
        assert!(response.degraded_reason.contains("dev stub"));

        let response = service
            .evaluate(Request::new(EvaluateRequest {
                project_id: "project-a".to_string(),
                policy_version: "1".to_string(),
                request_id: "req-2".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4o".to_string(),
                streaming: false,
                chunks: vec![Chunk {
                    path: "$.messages[0].content".to_string(),
                    text: "Something is happening at Company X, layoffs next week".to_string(),
                }],
                top_k: 3,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.findings.len(), 1);
        assert_eq!(response.findings[0].topic_id, "layoffs");
        assert_eq!(response.index_state, IndexState::Degraded as i32);
        assert!(response.degraded_reason.contains("dev stub"));
    }

    #[test]
    fn exact_alias_matching_rejects_substrings() {
        assert!(contains_exact_alias(
            "company x layoffs next week",
            "company x"
        ));
        assert!(contains_exact_alias(
            "company x, layoffs next week",
            "company x"
        ));
        assert!(!contains_exact_alias(
            "company xyz layoffs next week",
            "company x"
        ));
        assert!(!contains_exact_alias(
            "precompany x layoffs next week",
            "company x"
        ));
    }

    #[tokio::test]
    async fn health_reports_stub_backend_as_not_ready() {
        let dir = tempdir().unwrap();
        let service = SemanticSafetyGrpcService::new(
            SemanticSafetyConfig { auth_token: None },
            Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap()),
            Arc::new(TensorRtBackend::new_dev_stub()),
        );

        let response = service
            .health(Request::new(HealthRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert!(!response.ready);
        assert_eq!(response.backend, "tensorrt-dev-stub");
        assert!(response.message.contains("dev stub"));
    }

    #[tokio::test]
    async fn evaluate_returns_degraded_response_when_backend_fails() {
        let dir = tempdir().unwrap();
        let service = SemanticSafetyGrpcService::new(
            SemanticSafetyConfig { auth_token: None },
            Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap()),
            Arc::new(FailingBackend),
        );
        service
            .upsert_project_policy(Request::new(UpsertProjectPolicyRequest {
                policy: Some(sample_policy()),
            }))
            .await
            .unwrap();

        let response = service
            .evaluate(Request::new(EvaluateRequest {
                project_id: "project-a".to_string(),
                policy_version: "1".to_string(),
                request_id: "req-3".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4o".to_string(),
                streaming: false,
                chunks: vec![Chunk {
                    path: "$.messages[0].content".to_string(),
                    text: "Something is happening at Company X, layoffs next week".to_string(),
                }],
                top_k: 3,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.index_state, IndexState::Degraded as i32);
        assert!(response.findings.is_empty());
        assert!(response.degraded_reason.contains("rerank failed"));
    }

    #[tokio::test]
    async fn evaluate_returns_degraded_response_when_backend_embedding_count_mismatches() {
        let dir = tempdir().unwrap();
        let service = SemanticSafetyGrpcService::new(
            SemanticSafetyConfig { auth_token: None },
            Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap()),
            Arc::new(ShortEmbeddingBackend),
        );
        service
            .upsert_project_policy(Request::new(UpsertProjectPolicyRequest {
                policy: Some(sample_policy()),
            }))
            .await
            .unwrap();

        let response = service
            .evaluate(Request::new(EvaluateRequest {
                project_id: "project-a".to_string(),
                policy_version: "1".to_string(),
                request_id: "req-embedding-mismatch".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4o".to_string(),
                streaming: false,
                chunks: vec![
                    Chunk {
                        path: "$.messages[0].content".to_string(),
                        text: "Something is happening at Company X".to_string(),
                    },
                    Chunk {
                        path: "$.messages[1].content".to_string(),
                        text: "Layoffs next week".to_string(),
                    },
                ],
                top_k: 3,
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(response.index_state, IndexState::Degraded as i32);
        assert!(response.findings.is_empty());
        assert!(response
            .degraded_reason
            .contains("embedding count mismatch"));
    }

    #[tokio::test]
    async fn load_from_store_reuses_persisted_exemplar_embeddings() {
        let dir = tempdir().unwrap();
        let store = Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap());
        let initial_backend = Arc::new(CountingBackend::default());
        let service = SemanticSafetyGrpcService::new(
            SemanticSafetyConfig { auth_token: None },
            Arc::clone(&store),
            initial_backend.clone(),
        );

        service
            .upsert_project_policy(Request::new(UpsertProjectPolicyRequest {
                policy: Some(sample_policy()),
            }))
            .await
            .unwrap();
        assert_eq!(initial_backend.embed_calls.load(AtomicOrdering::SeqCst), 1);

        let reload_backend = Arc::new(CountingBackend::default());
        let reloaded_service = SemanticSafetyGrpcService::new(
            SemanticSafetyConfig { auth_token: None },
            Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap()),
            reload_backend.clone(),
        );
        reloaded_service.load_from_store().await.unwrap();

        assert_eq!(reload_backend.embed_calls.load(AtomicOrdering::SeqCst), 0);

        let response = reloaded_service
            .evaluate(Request::new(EvaluateRequest {
                project_id: "project-a".to_string(),
                policy_version: "1".to_string(),
                request_id: "req-reload".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4o".to_string(),
                streaming: false,
                chunks: vec![Chunk {
                    path: "$.messages[0].content".to_string(),
                    text: "Something is happening at Company X, layoffs next week".to_string(),
                }],
                top_k: 3,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.index_state, IndexState::Ready as i32);
        assert_eq!(response.findings.len(), 1);
        assert_eq!(reload_backend.rerank_calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn metrics_track_index_and_evaluate_outcomes() {
        let dir = tempdir().unwrap();
        let metrics = Arc::new(SemanticSafetyServiceMetrics::new());
        let service = SemanticSafetyGrpcService::new(
            SemanticSafetyConfig { auth_token: None },
            Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap()),
            Arc::new(TensorRtBackend::new_dev_stub()),
        )
        .with_metrics(Arc::clone(&metrics));

        service
            .upsert_project_policy(Request::new(UpsertProjectPolicyRequest {
                policy: Some(sample_policy()),
            }))
            .await
            .unwrap();
        assert_eq!(metrics.indexed_projects.get(), 1);
        assert_eq!(metrics.indexed_exemplars.get(), 1);

        let response = service
            .evaluate(Request::new(EvaluateRequest {
                project_id: "project-a".to_string(),
                policy_version: "1".to_string(),
                request_id: "req-metrics".to_string(),
                path: "/v1/chat/completions".to_string(),
                model: "gpt-4o".to_string(),
                streaming: false,
                chunks: vec![Chunk {
                    path: "$.messages[0].content".to_string(),
                    text: "Something is happening at Company X, layoffs next week".to_string(),
                }],
                top_k: 3,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.index_state, IndexState::Degraded as i32);

        let rendered = metrics.render().unwrap();
        assert!(rendered.contains("semantic_safety_service_evaluate_requests_total"));
        assert!(rendered.contains("outcome=\"degraded\""));
        assert!(rendered.contains("semantic_safety_service_indexed_projects 1"));
    }
}
