#![allow(non_camel_case_types)]

use std::ffi::c_char;
#[cfg(not(feature = "native-tensorrt"))]
use std::ffi::c_void;

#[repr(C)]
pub struct ss_trt_status {
    pub code: i32,
    pub message: *mut c_char,
}

#[repr(C)]
pub struct ss_trt_embedding_engine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ss_trt_reranker_engine {
    _private: [u8; 0],
}

#[cfg(feature = "native-tensorrt")]
extern "C" {
    pub fn ss_trt_status_free(status: *mut ss_trt_status);

    pub fn ss_trt_embedding_engine_load(
        engine_path: *const c_char,
        device_id: i32,
        max_batch_size: usize,
        max_sequence_length: usize,
        status: *mut ss_trt_status,
    ) -> *mut ss_trt_embedding_engine;
    pub fn ss_trt_embedding_engine_destroy(engine: *mut ss_trt_embedding_engine);
    pub fn ss_trt_embedding_engine_output_dim(engine: *const ss_trt_embedding_engine) -> usize;
    pub fn ss_trt_embedding_engine_warmup(
        engine: *mut ss_trt_embedding_engine,
        status: *mut ss_trt_status,
    ) -> bool;
    pub fn ss_trt_embedding_engine_infer(
        engine: *mut ss_trt_embedding_engine,
        input_ids: *const i32,
        attention_mask: *const i32,
        batch_size: usize,
        sequence_length: usize,
        output_embeddings: *mut f32,
        output_len: usize,
        status: *mut ss_trt_status,
    ) -> bool;

    pub fn ss_trt_reranker_engine_load(
        engine_path: *const c_char,
        device_id: i32,
        max_batch_size: usize,
        max_sequence_length: usize,
        status: *mut ss_trt_status,
    ) -> *mut ss_trt_reranker_engine;
    pub fn ss_trt_reranker_engine_destroy(engine: *mut ss_trt_reranker_engine);
    pub fn ss_trt_reranker_engine_warmup(
        engine: *mut ss_trt_reranker_engine,
        status: *mut ss_trt_status,
    ) -> bool;
    pub fn ss_trt_reranker_engine_infer(
        engine: *mut ss_trt_reranker_engine,
        input_ids: *const i32,
        attention_mask: *const i32,
        batch_size: usize,
        sequence_length: usize,
        output_scores: *mut f32,
        output_len: usize,
        status: *mut ss_trt_status,
    ) -> bool;
}

#[cfg(not(feature = "native-tensorrt"))]
#[allow(dead_code)]
pub type ss_trt_placeholder = c_void;
