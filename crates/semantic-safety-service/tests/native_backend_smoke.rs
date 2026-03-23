#![cfg(feature = "native-tensorrt")]

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use semantic_safety_protocol::{
    Chunk, EvaluateRequest, IndexState, ProjectSemanticPolicy, SemanticEntity,
    SemanticSafetyService, SemanticTopic, UpsertProjectPolicyRequest,
};
use semantic_safety_service::backend::{InferenceBackend, TensorRtBackend, TensorRtBackendConfig};
use semantic_safety_service::persistence::FileProjectIndexStore;
use semantic_safety_service::service::{SemanticSafetyConfig, SemanticSafetyGrpcService};
use tempfile::tempdir;
use tonic::Request;

fn required_env_path(name: &str) -> PathBuf {
    let value = env::var(name).unwrap_or_else(|_| {
        panic!(
            "{name} is required for native TensorRT smoke tests; point it at a real semantic safety asset path"
        )
    });
    let trimmed = value.trim();
    assert!(!trimmed.is_empty(), "{name} must not be empty");
    PathBuf::from(trimmed)
}

fn env_or_default<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn native_test_config() -> TensorRtBackendConfig {
    TensorRtBackendConfig {
        embedding_engine: required_env_path("SEMANTIC_SAFETY_EMBEDDING_ENGINE"),
        reranker_engine: required_env_path("SEMANTIC_SAFETY_RERANKER_ENGINE"),
        tokenizer_dir: required_env_path("SEMANTIC_SAFETY_TOKENIZER_DIR"),
        device_id: env_or_default("SEMANTIC_SAFETY_DEVICE_ID", 0i32),
        max_batch_size: env_or_default("SEMANTIC_SAFETY_MAX_BATCH_SIZE", 8usize),
        max_sequence_length: env_or_default("SEMANTIC_SAFETY_MAX_SEQUENCE_LENGTH", 512usize),
        warmup_enabled: true,
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| l * r)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm * right_norm)
}

fn sample_policy() -> ProjectSemanticPolicy {
    ProjectSemanticPolicy {
        project_id: "native-smoke".to_string(),
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
            rerank_threshold: 0.01,
            require_entity_match: true,
        }],
        updated_at: "1".to_string(),
    }
}

#[tokio::test]
#[ignore = "requires a native-tensorrt build plus real TensorRT engines/tokenizer assets"]
async fn native_backend_reports_ready_and_prefers_related_embeddings() {
    let backend = TensorRtBackend::from_config(native_test_config()).unwrap();
    let health = backend.health();
    assert!(health.ready);
    assert_eq!(health.backend, "tensorrt-native");

    let embeddings = backend
        .embed_texts(&[
            "Company X layoffs next week".to_string(),
            "Memo about Company X layoffs".to_string(),
            "Banana bread recipe".to_string(),
        ])
        .await
        .unwrap();
    assert_eq!(embeddings.len(), 3);
    assert!(embeddings
        .iter()
        .flat_map(|embedding| embedding.iter())
        .all(|value| value.is_finite()));

    let related = cosine_similarity(&embeddings[0], &embeddings[1]);
    let unrelated = cosine_similarity(&embeddings[0], &embeddings[2]);
    assert!(
        related > unrelated,
        "expected related texts to be closer than unrelated ones ({related} <= {unrelated})"
    );
}

#[tokio::test]
#[ignore = "requires a native-tensorrt build plus real TensorRT engines/tokenizer assets"]
async fn native_backend_reranker_prefers_relevant_candidate() {
    let backend = TensorRtBackend::from_config(native_test_config()).unwrap();
    let scores = backend
        .rerank(
            "company x layoffs",
            &[
                "Company X layoffs were announced internally.".to_string(),
                "Banana bread tips and baking times.".to_string(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(scores.len(), 2);
    assert!(
        scores[0] > scores[1],
        "expected relevant candidate to score higher ({:?})",
        scores
    );
}

#[tokio::test]
#[ignore = "requires a native-tensorrt build plus real TensorRT engines/tokenizer assets"]
async fn native_service_evaluates_shadow_mode_request() {
    let dir = tempdir().unwrap();
    let service = SemanticSafetyGrpcService::new(
        SemanticSafetyConfig { auth_token: None },
        Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap()),
        Arc::new(TensorRtBackend::from_config(native_test_config()).unwrap()),
    );

    service
        .upsert_project_policy(Request::new(UpsertProjectPolicyRequest {
            policy: Some(sample_policy()),
        }))
        .await
        .unwrap();

    let response = service
        .evaluate(Request::new(EvaluateRequest {
            project_id: "native-smoke".to_string(),
            policy_version: "1".to_string(),
            request_id: "req-native".to_string(),
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
    assert!(
        !response.findings.is_empty(),
        "expected at least one semantic finding from the native backend"
    );
}
