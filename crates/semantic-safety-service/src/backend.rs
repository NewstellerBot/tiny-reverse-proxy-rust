use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use semantic_safety_trt::{
    NativeEmbeddingEngine, NativeRerankerEngine, QwenTokenizer, TensorRtModelConfig,
    DEFAULT_EMBEDDING_INSTRUCTION, DEFAULT_RERANK_INSTRUCTION,
};

#[cfg(any(test, feature = "dev_stub_backend"))]
use std::collections::HashSet;

type BackendError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendHealth {
    pub ready: bool,
    pub backend: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRtBackendConfig {
    pub embedding_engine: PathBuf,
    pub reranker_engine: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub device_id: i32,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub warmup_enabled: bool,
}

impl TensorRtBackendConfig {
    pub fn from_env() -> Result<Self, BackendError> {
        let embedding_engine = required_path("SEMANTIC_SAFETY_EMBEDDING_ENGINE")?;
        let reranker_engine = required_path("SEMANTIC_SAFETY_RERANKER_ENGINE")?;
        let tokenizer_dir = required_path("SEMANTIC_SAFETY_TOKENIZER_DIR")?;
        let device_id = env::var("SEMANTIC_SAFETY_DEVICE_ID")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(0);
        let max_batch_size = env::var("SEMANTIC_SAFETY_MAX_BATCH_SIZE")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(8usize);
        let max_sequence_length = env::var("SEMANTIC_SAFETY_MAX_SEQUENCE_LENGTH")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(512usize);
        let warmup_enabled = env::var("SEMANTIC_SAFETY_WARMUP_ENABLED")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(true);

        let config = Self {
            embedding_engine,
            reranker_engine,
            tokenizer_dir,
            device_id,
            max_batch_size,
            max_sequence_length,
            warmup_enabled,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        let model = self.model_config();
        model.validate()?;
        if self.max_batch_size == 0 {
            return Err("SEMANTIC_SAFETY_MAX_BATCH_SIZE must be greater than 0".into());
        }
        if self.max_sequence_length == 0 {
            return Err("SEMANTIC_SAFETY_MAX_SEQUENCE_LENGTH must be greater than 0".into());
        }
        Ok(())
    }

    pub fn model_config(&self) -> TensorRtModelConfig {
        TensorRtModelConfig {
            embedding_engine: self.embedding_engine.clone(),
            reranker_engine: self.reranker_engine.clone(),
            tokenizer_dir: self.tokenizer_dir.clone(),
            device_id: self.device_id,
            max_batch_size: self.max_batch_size,
            max_sequence_length: self.max_sequence_length,
        }
    }
}

fn required_path(name: &str) -> Result<PathBuf, BackendError> {
    let value = env::var(name).map_err(|_| format!("{name} is required"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{name} must not be empty").into());
    }
    Ok(PathBuf::from(trimmed))
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, BackendError>;

    async fn rerank(&self, query: &str, candidates: &[String]) -> Result<Vec<f32>, BackendError>;

    fn backend_name(&self) -> &'static str;

    fn health(&self) -> BackendHealth;
}

struct NativeState {
    tokenizer: QwenTokenizer,
    embedding_engine: Mutex<NativeEmbeddingEngine>,
    reranker_engine: Mutex<NativeRerankerEngine>,
    max_batch_size: usize,
    max_sequence_length: usize,
    health: BackendHealth,
}

#[derive(Clone)]
pub struct TensorRtBackend {
    kind: Arc<BackendKind>,
}

enum BackendKind {
    Native(Box<NativeState>),
    #[cfg(any(test, feature = "dev_stub_backend"))]
    DevStub,
}

impl TensorRtBackend {
    pub fn from_env() -> Result<Self, BackendError> {
        Self::from_config(TensorRtBackendConfig::from_env()?)
    }

    pub fn from_config(config: TensorRtBackendConfig) -> Result<Self, BackendError> {
        config.validate()?;

        let tokenizer = QwenTokenizer::from_dir(&config.tokenizer_dir)?;
        let model = config.model_config();
        let mut embedding_engine = NativeEmbeddingEngine::load(&model)?;
        let mut reranker_engine = NativeRerankerEngine::load(&model)?;
        let health = if config.warmup_enabled {
            embedding_engine.warmup()?;
            reranker_engine.warmup()?;
            BackendHealth {
                ready: true,
                backend: "tensorrt-native",
                message: "native TensorRT engines loaded and warmup completed".to_string(),
            }
        } else {
            BackendHealth {
                ready: true,
                backend: "tensorrt-native",
                message: "native TensorRT engines loaded; warmup disabled".to_string(),
            }
        };

        Ok(Self {
            kind: Arc::new(BackendKind::Native(Box::new(NativeState {
                tokenizer,
                embedding_engine: Mutex::new(embedding_engine),
                reranker_engine: Mutex::new(reranker_engine),
                max_batch_size: config.max_batch_size,
                max_sequence_length: config.max_sequence_length,
                health,
            }))),
        })
    }

    #[cfg(any(test, feature = "dev_stub_backend"))]
    pub fn new_dev_stub() -> Self {
        Self {
            kind: Arc::new(BackendKind::DevStub),
        }
    }

    #[cfg(any(test, feature = "dev_stub_backend"))]
    fn overlap_score(query: &str, candidate: &str) -> f32 {
        let query_tokens: HashSet<String> = query
            .split_whitespace()
            .map(|token| token.to_ascii_lowercase())
            .collect();
        let candidate_tokens: HashSet<String> = candidate
            .split_whitespace()
            .map(|token| token.to_ascii_lowercase())
            .collect();
        if query_tokens.is_empty() || candidate_tokens.is_empty() {
            return 0.0;
        }
        let overlap = query_tokens.intersection(&candidate_tokens).count() as f32;
        overlap / query_tokens.len().max(candidate_tokens.len()) as f32
    }

    #[cfg(any(test, feature = "dev_stub_backend"))]
    fn hashed_embedding(text: &str) -> Vec<f32> {
        use sha2::{Digest, Sha256};

        let mut vector = vec![0.0f32; 64];
        for token in text.split_whitespace() {
            let mut hasher = Sha256::new();
            hasher.update(token.as_bytes());
            let digest = hasher.finalize();
            for (idx, value) in vector.iter_mut().enumerate() {
                let byte = digest[idx % digest.len()] as f32 / 255.0;
                *value += byte;
            }
        }
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        vector
    }
}

#[async_trait]
impl InferenceBackend for TensorRtBackend {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, BackendError> {
        match self.kind.as_ref() {
            BackendKind::Native(state) => {
                let mut output = Vec::new();
                for chunk in texts.chunks(state.max_batch_size) {
                    let batch = state.tokenizer.encode_embedding_queries(
                        DEFAULT_EMBEDDING_INSTRUCTION,
                        chunk,
                        state.max_sequence_length,
                    )?;
                    let mut engine = state
                        .embedding_engine
                        .lock()
                        .map_err(|_| "embedding engine mutex poisoned")?;
                    output.extend(engine.infer(&batch)?);
                }
                Ok(output)
            }
            #[cfg(any(test, feature = "dev_stub_backend"))]
            BackendKind::DevStub => Ok(texts
                .iter()
                .map(|text| Self::hashed_embedding(text))
                .collect()),
        }
    }

    async fn rerank(&self, query: &str, candidates: &[String]) -> Result<Vec<f32>, BackendError> {
        match self.kind.as_ref() {
            BackendKind::Native(state) => {
                let mut output = Vec::new();
                let query = query.to_string();
                for chunk in candidates.chunks(state.max_batch_size) {
                    let pairs = chunk
                        .iter()
                        .map(|candidate| (query.clone(), candidate.clone()))
                        .collect::<Vec<_>>();
                    let batch = state.tokenizer.encode_reranker_pairs(
                        DEFAULT_RERANK_INSTRUCTION,
                        &pairs,
                        state.max_sequence_length,
                    )?;
                    let mut engine = state
                        .reranker_engine
                        .lock()
                        .map_err(|_| "reranker engine mutex poisoned")?;
                    output.extend(engine.infer(&batch)?);
                }
                Ok(output)
            }
            #[cfg(any(test, feature = "dev_stub_backend"))]
            BackendKind::DevStub => Ok(candidates
                .iter()
                .map(|candidate| Self::overlap_score(query, candidate))
                .collect()),
        }
    }

    fn backend_name(&self) -> &'static str {
        match self.kind.as_ref() {
            BackendKind::Native(state) => state.health.backend,
            #[cfg(any(test, feature = "dev_stub_backend"))]
            BackendKind::DevStub => "tensorrt-dev-stub",
        }
    }

    fn health(&self) -> BackendHealth {
        match self.kind.as_ref() {
            BackendKind::Native(state) => state.health.clone(),
            #[cfg(any(test, feature = "dev_stub_backend"))]
            BackendKind::DevStub => BackendHealth {
                ready: false,
                backend: "tensorrt-dev-stub",
                message: "dev stub backend enabled; production builds must use native TensorRT"
                    .into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn config_requires_assets() {
        let dir = tempfile::tempdir().unwrap();
        let config = TensorRtBackendConfig {
            embedding_engine: dir.path().join("embedding.engine"),
            reranker_engine: dir.path().join("reranker.engine"),
            tokenizer_dir: dir.path().join("tokenizer"),
            device_id: 0,
            max_batch_size: 8,
            max_sequence_length: 256,
            warmup_enabled: true,
        };
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("embedding engine"));
    }

    #[tokio::test]
    async fn dev_stub_backend_returns_local_scores() {
        let backend = TensorRtBackend::new_dev_stub();
        let embeddings = backend
            .embed_texts(&["company x".to_string(), "layoffs next week".to_string()])
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(
            backend.health(),
            BackendHealth {
                ready: false,
                backend: "tensorrt-dev-stub",
                message: "dev stub backend enabled; production builds must use native TensorRT"
                    .to_string(),
            }
        );

        let scores = backend
            .rerank(
                "company x layoffs",
                &["company x layoffs".to_string(), "hello world".to_string()],
            )
            .await
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn config_validation_passes_with_fake_assets() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer_dir = dir.path().join("tokenizer");
        fs::create_dir_all(&tokenizer_dir).unwrap();
        fs::write(dir.path().join("embedding.engine"), b"engine").unwrap();
        fs::write(dir.path().join("reranker.engine"), b"engine").unwrap();
        fs::write(tokenizer_dir.join("tokenizer.json"), b"{}").unwrap();

        let config = TensorRtBackendConfig {
            embedding_engine: dir.path().join("embedding.engine"),
            reranker_engine: dir.path().join("reranker.engine"),
            tokenizer_dir,
            device_id: 0,
            max_batch_size: 8,
            max_sequence_length: 256,
            warmup_enabled: false,
        };
        config.validate().unwrap();
    }
}
