use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

type TrtError = Box<dyn std::error::Error + Send + Sync>;

pub const DEFAULT_EMBEDDING_INSTRUCTION: &str =
    "Given a request text chunk, retrieve semantic safety topic exemplars that describe the same business-sensitive situation.";
pub const DEFAULT_RERANK_INSTRUCTION: &str =
    "Given a request text chunk, retrieve semantic safety topic exemplars that describe the same business-sensitive situation.";

const RERANKER_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const RERANKER_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

#[derive(Clone, Debug)]
pub struct TensorRtModelConfig {
    pub embedding_engine: PathBuf,
    pub reranker_engine: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub device_id: i32,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
}

impl TensorRtModelConfig {
    pub fn validate(&self) -> Result<(), TrtError> {
        if !self.embedding_engine.exists() {
            return Err(format!(
                "embedding engine does not exist: {}",
                self.embedding_engine.display()
            )
            .into());
        }
        if !self.reranker_engine.exists() {
            return Err(format!(
                "reranker engine does not exist: {}",
                self.reranker_engine.display()
            )
            .into());
        }
        if !self.tokenizer_dir.join("tokenizer.json").exists() {
            return Err(format!(
                "tokenizer.json does not exist in {}",
                self.tokenizer_dir.display()
            )
            .into());
        }
        if self.max_batch_size == 0 {
            return Err("max batch size must be greater than 0".into());
        }
        if self.max_sequence_length == 0 {
            return Err("max sequence length must be greater than 0".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedBatch {
    pub input_ids: Vec<i32>,
    pub attention_mask: Vec<i32>,
    pub batch_size: usize,
    pub sequence_length: usize,
}

#[cfg(any(test, feature = "native-tensorrt"))]
fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

#[derive(Clone)]
pub struct QwenTokenizer {
    tokenizer: Tokenizer,
    pad_id: u32,
}

impl QwenTokenizer {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, TrtError> {
        let tokenizer_path = dir.as_ref().join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| format!("failed to load tokenizer.json: {error}"))?;
        let pad_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|padding|>"))
            .ok_or("tokenizer is missing an end-of-text or padding token")?;
        Ok(Self { tokenizer, pad_id })
    }

    #[cfg(test)]
    fn from_parts(tokenizer: Tokenizer, pad_id: u32) -> Self {
        Self { tokenizer, pad_id }
    }

    pub fn encode_embedding_documents(
        &self,
        texts: &[String],
        max_length: usize,
    ) -> Result<EncodedBatch, TrtError> {
        self.encode_plain_batch(texts, max_length)
    }

    pub fn encode_embedding_queries(
        &self,
        instruction: &str,
        texts: &[String],
        max_length: usize,
    ) -> Result<EncodedBatch, TrtError> {
        let formatted = texts
            .iter()
            .map(|text| format!("Instruct: {instruction}\nQuery:{text}"))
            .collect::<Vec<_>>();
        self.encode_plain_batch(&formatted, max_length)
    }

    pub fn encode_reranker_pairs(
        &self,
        instruction: &str,
        pairs: &[(String, String)],
        max_length: usize,
    ) -> Result<EncodedBatch, TrtError> {
        let prefix = self
            .tokenizer
            .encode(RERANKER_PREFIX, false)
            .map_err(|error| format!("failed to encode reranker prefix: {error}"))?;
        let suffix = self
            .tokenizer
            .encode(RERANKER_SUFFIX, false)
            .map_err(|error| format!("failed to encode reranker suffix: {error}"))?;
        let prefix_ids = prefix.get_ids().to_vec();
        let suffix_ids = suffix.get_ids().to_vec();
        if prefix_ids.len() + suffix_ids.len() >= max_length {
            return Err("max sequence length is too small for reranker prefix/suffix".into());
        }

        let body_len = max_length - prefix_ids.len() - suffix_ids.len();
        let mut rows = Vec::with_capacity(pairs.len());
        for (query, document) in pairs {
            let encoded = self
                .tokenizer
                .encode(
                    format!("<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"),
                    false,
                )
                .map_err(|error| format!("failed to encode reranker pair: {error}"))?;
            let mut body = encoded.get_ids().to_vec();
            if body.len() > body_len {
                body.truncate(body_len);
            }
            let mut row = Vec::with_capacity(max_length);
            row.extend_from_slice(&prefix_ids);
            row.extend_from_slice(&body);
            row.extend_from_slice(&suffix_ids);
            rows.push(row);
        }
        self.left_pad_batch(rows, max_length)
    }

    fn encode_plain_batch(
        &self,
        texts: &[String],
        max_length: usize,
    ) -> Result<EncodedBatch, TrtError> {
        let mut rows = Vec::with_capacity(texts.len());
        for text in texts {
            let encoded = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|error| format!("failed to encode text: {error}"))?;
            let mut row = encoded.get_ids().to_vec();
            if row.len() > max_length {
                row.truncate(max_length);
            }
            rows.push(row);
        }
        self.left_pad_batch(rows, max_length)
    }

    fn left_pad_batch(
        &self,
        rows: Vec<Vec<u32>>,
        max_length: usize,
    ) -> Result<EncodedBatch, TrtError> {
        let batch_size = rows.len();
        let mut input_ids = Vec::with_capacity(batch_size * max_length);
        let mut attention_mask = Vec::with_capacity(batch_size * max_length);
        for row in rows {
            if row.len() > max_length {
                return Err("token row exceeded max length after truncation".into());
            }
            let pad_len = max_length - row.len();
            input_ids.extend(std::iter::repeat(self.pad_id as i32).take(pad_len));
            attention_mask.extend(std::iter::repeat(0).take(pad_len));
            input_ids.extend(row.iter().map(|token| *token as i32));
            attention_mask.extend(std::iter::repeat(1).take(row.len()));
        }
        Ok(EncodedBatch {
            input_ids,
            attention_mask,
            batch_size,
            sequence_length: max_length,
        })
    }
}

#[cfg(feature = "native-tensorrt")]
mod native {
    use super::*;
    use semantic_safety_trt_sys::{
        ss_trt_embedding_engine, ss_trt_embedding_engine_destroy, ss_trt_embedding_engine_infer,
        ss_trt_embedding_engine_load, ss_trt_embedding_engine_output_dim,
        ss_trt_embedding_engine_warmup, ss_trt_reranker_engine, ss_trt_reranker_engine_destroy,
        ss_trt_reranker_engine_infer, ss_trt_reranker_engine_load, ss_trt_reranker_engine_warmup,
        ss_trt_status, ss_trt_status_free,
    };
    use std::ffi::{CStr, CString};

    struct StatusGuard {
        raw: ss_trt_status,
    }

    impl Default for StatusGuard {
        fn default() -> Self {
            Self {
                raw: ss_trt_status {
                    code: 0,
                    message: std::ptr::null_mut(),
                },
            }
        }
    }

    impl StatusGuard {
        fn into_result(mut self) -> Result<(), TrtError> {
            if self.raw.code == 0 {
                return Ok(());
            }
            let message = unsafe {
                if self.raw.message.is_null() {
                    "unknown TensorRT error".to_string()
                } else {
                    CStr::from_ptr(self.raw.message)
                        .to_string_lossy()
                        .to_string()
                }
            };
            unsafe { ss_trt_status_free(&mut self.raw) }
            Err(message.into())
        }
    }

    impl Drop for StatusGuard {
        fn drop(&mut self) {
            unsafe { ss_trt_status_free(&mut self.raw) }
        }
    }

    pub struct EmbeddingEngine {
        raw: *mut ss_trt_embedding_engine,
        output_dim: usize,
    }

    // The raw TensorRT handle is only accessed through owned methods and is
    // externally serialized by the caller when shared across threads.
    unsafe impl Send for EmbeddingEngine {}

    impl EmbeddingEngine {
        pub fn load(config: &TensorRtModelConfig) -> Result<Self, TrtError> {
            config.validate()?;
            let engine_path = CString::new(config.embedding_engine.display().to_string())?;
            let mut status = StatusGuard::default();
            let raw = unsafe {
                ss_trt_embedding_engine_load(
                    engine_path.as_ptr(),
                    config.device_id,
                    config.max_batch_size,
                    config.max_sequence_length,
                    &mut status.raw,
                )
            };
            if raw.is_null() {
                return status
                    .into_result()
                    .and(Err("failed to load embedding engine".into()));
            }
            let output_dim = unsafe { ss_trt_embedding_engine_output_dim(raw) };
            status.into_result()?;
            Ok(Self { raw, output_dim })
        }

        pub fn output_dim(&self) -> usize {
            self.output_dim
        }

        pub fn warmup(&mut self) -> Result<(), TrtError> {
            let mut status = StatusGuard::default();
            let ok = unsafe { ss_trt_embedding_engine_warmup(self.raw, &mut status.raw) };
            status.into_result()?;
            if !ok {
                return Err("embedding engine warmup failed".into());
            }
            Ok(())
        }

        pub fn infer(&mut self, batch: &EncodedBatch) -> Result<Vec<Vec<f32>>, TrtError> {
            let output_len = batch.batch_size * self.output_dim;
            let mut output = vec![0.0f32; output_len];
            let mut status = StatusGuard::default();
            let ok = unsafe {
                ss_trt_embedding_engine_infer(
                    self.raw,
                    batch.input_ids.as_ptr(),
                    batch.attention_mask.as_ptr(),
                    batch.batch_size,
                    batch.sequence_length,
                    output.as_mut_ptr(),
                    output.len(),
                    &mut status.raw,
                )
            };
            status.into_result()?;
            if !ok {
                return Err("embedding engine inference failed".into());
            }
            Ok(output
                .chunks(self.output_dim)
                .map(|chunk| {
                    let mut vector = chunk.to_vec();
                    l2_normalize(&mut vector);
                    vector
                })
                .collect())
        }
    }

    impl Drop for EmbeddingEngine {
        fn drop(&mut self) {
            unsafe { ss_trt_embedding_engine_destroy(self.raw) }
        }
    }

    pub struct RerankerEngine {
        raw: *mut ss_trt_reranker_engine,
    }

    // The raw TensorRT handle is only accessed through owned methods and is
    // externally serialized by the caller when shared across threads.
    unsafe impl Send for RerankerEngine {}

    impl RerankerEngine {
        pub fn load(config: &TensorRtModelConfig) -> Result<Self, TrtError> {
            config.validate()?;
            let engine_path = CString::new(config.reranker_engine.display().to_string())?;
            let mut status = StatusGuard::default();
            let raw = unsafe {
                ss_trt_reranker_engine_load(
                    engine_path.as_ptr(),
                    config.device_id,
                    config.max_batch_size,
                    config.max_sequence_length,
                    &mut status.raw,
                )
            };
            if raw.is_null() {
                return status
                    .into_result()
                    .and(Err("failed to load reranker engine".into()));
            }
            status.into_result()?;
            Ok(Self { raw })
        }

        pub fn warmup(&mut self) -> Result<(), TrtError> {
            let mut status = StatusGuard::default();
            let ok = unsafe { ss_trt_reranker_engine_warmup(self.raw, &mut status.raw) };
            status.into_result()?;
            if !ok {
                return Err("reranker engine warmup failed".into());
            }
            Ok(())
        }

        pub fn infer(&mut self, batch: &EncodedBatch) -> Result<Vec<f32>, TrtError> {
            let mut output = vec![0.0f32; batch.batch_size];
            let mut status = StatusGuard::default();
            let ok = unsafe {
                ss_trt_reranker_engine_infer(
                    self.raw,
                    batch.input_ids.as_ptr(),
                    batch.attention_mask.as_ptr(),
                    batch.batch_size,
                    batch.sequence_length,
                    output.as_mut_ptr(),
                    output.len(),
                    &mut status.raw,
                )
            };
            status.into_result()?;
            if !ok {
                return Err("reranker engine inference failed".into());
            }
            Ok(output)
        }
    }

    impl Drop for RerankerEngine {
        fn drop(&mut self) {
            unsafe { ss_trt_reranker_engine_destroy(self.raw) }
        }
    }

    pub use EmbeddingEngine as NativeEmbeddingEngine;
    pub use RerankerEngine as NativeRerankerEngine;
}

#[cfg(not(feature = "native-tensorrt"))]
mod native {
    use super::*;

    pub struct EmbeddingEngine;
    pub struct RerankerEngine;

    impl EmbeddingEngine {
        pub fn load(_config: &TensorRtModelConfig) -> Result<Self, TrtError> {
            Err("native TensorRT support is not compiled in; rebuild semantic-safety-service with --features native-tensorrt".into())
        }

        pub fn output_dim(&self) -> usize {
            0
        }

        pub fn warmup(&mut self) -> Result<(), TrtError> {
            Err("native TensorRT support is not compiled in".into())
        }

        pub fn infer(&mut self, _batch: &EncodedBatch) -> Result<Vec<Vec<f32>>, TrtError> {
            Err("native TensorRT support is not compiled in".into())
        }
    }

    impl RerankerEngine {
        pub fn load(_config: &TensorRtModelConfig) -> Result<Self, TrtError> {
            Err("native TensorRT support is not compiled in; rebuild semantic-safety-service with --features native-tensorrt".into())
        }

        pub fn warmup(&mut self) -> Result<(), TrtError> {
            Err("native TensorRT support is not compiled in".into())
        }

        pub fn infer(&mut self, _batch: &EncodedBatch) -> Result<Vec<f32>, TrtError> {
            Err("native TensorRT support is not compiled in".into())
        }
    }

    pub use EmbeddingEngine as NativeEmbeddingEngine;
    pub use RerankerEngine as NativeRerankerEngine;
}

pub use native::{NativeEmbeddingEngine, NativeRerankerEngine};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use super::*;

    fn test_tokenizer() -> QwenTokenizer {
        let vocab = HashMap::from([
            ("[UNK]".to_string(), 0u32),
            ("<|endoftext|>".to_string(), 1u32),
            ("Instruct".to_string(), 2u32),
            (":".to_string(), 3u32),
            ("Query".to_string(), 4u32),
            ("Given".to_string(), 5u32),
            ("a".to_string(), 6u32),
            ("request".to_string(), 7u32),
            ("text".to_string(), 8u32),
            ("chunk".to_string(), 9u32),
            ("retrieve".to_string(), 10u32),
            ("semantic".to_string(), 11u32),
            ("safety".to_string(), 12u32),
            ("topic".to_string(), 13u32),
            ("exemplars".to_string(), 14u32),
            ("that".to_string(), 15u32),
            ("describe".to_string(), 16u32),
            ("the".to_string(), 17u32),
            ("same".to_string(), 18u32),
            ("business".to_string(), 19u32),
            ("sensitive".to_string(), 20u32),
            ("situation".to_string(), 21u32),
            ("Company".to_string(), 22u32),
            ("X".to_string(), 23u32),
            ("layoffs".to_string(), 24u32),
            ("Document".to_string(), 25u32),
            ("Judge".to_string(), 26u32),
            ("whether".to_string(), 27u32),
            ("following".to_string(), 28u32),
            ("document".to_string(), 29u32),
            ("matches".to_string(), 30u32),
            ("query".to_string(), 31u32),
            ("in".to_string(), 32u32),
            ("context".to_string(), 33u32),
            ("of".to_string(), 34u32),
            ("policy".to_string(), 35u32),
            ("retrieval".to_string(), 36u32),
            ("Reply".to_string(), 37u32),
            ("with".to_string(), 38u32),
            ("only".to_string(), 39u32),
            ("yes".to_string(), 40u32),
            ("or".to_string(), 41u32),
            ("no".to_string(), 42u32),
            ("im_start".to_string(), 43u32),
            ("im_end".to_string(), 44u32),
            ("system".to_string(), 45u32),
            ("user".to_string(), 46u32),
            ("assistant".to_string(), 47u32),
        ]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
        QwenTokenizer::from_parts(tokenizer, 1)
    }

    #[test]
    fn embedding_queries_include_instruction_prefix() {
        let tokenizer = test_tokenizer();
        let batch = tokenizer
            .encode_embedding_queries(
                DEFAULT_EMBEDDING_INSTRUCTION,
                &["Company X layoffs".to_string()],
                16,
            )
            .unwrap();
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.sequence_length, 16);
        assert_eq!(batch.input_ids.len(), 16);
        let active_tokens = batch.attention_mask.iter().sum::<i32>();
        assert!(active_tokens > 0);
        assert!(active_tokens <= 16);
    }

    #[test]
    fn reranker_pairs_apply_prefix_and_suffix_wrapping() {
        let tokenizer = test_tokenizer();
        let batch = tokenizer
            .encode_reranker_pairs(
                DEFAULT_RERANK_INSTRUCTION,
                &[(
                    "Company X layoffs".to_string(),
                    "Company X document".to_string(),
                )],
                96,
            )
            .unwrap();
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.sequence_length, 96);
        assert_eq!(batch.input_ids.len(), 96);
        assert!(batch.attention_mask.iter().sum::<i32>() > 8);
    }

    #[test]
    fn l2_normalize_leaves_zero_vector_stable() {
        let mut vector = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut vector);
        assert_eq!(vector, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_scales_non_zero_vector() {
        let mut vector = vec![3.0f32, 4.0];
        l2_normalize(&mut vector);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }
}
