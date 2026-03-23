# Semantic Safety V0

Semantic safety v0 is a shadow-mode deployment that keeps the gateway as the
control plane and runs semantic evaluation in a separate Rust service on a GPU
host. The service performs in-process TensorRT inference for
`Qwen3-Embedding-0.6B` and `Qwen3-Reranker-0.6B`.

## Build the service binary

Build the Rust service with native TensorRT support enabled:

```bash
cargo build -p semantic-safety-service --release --features native-tensorrt
```

If TensorRT or CUDA are not installed in standard system include/library paths,
set these build-time environment variables before compiling:

- `SEMANTIC_SAFETY_TENSORRT_INCLUDE`
- `SEMANTIC_SAFETY_TENSORRT_LIB`
- `SEMANTIC_SAFETY_CUDA_INCLUDE`
- `SEMANTIC_SAFETY_CUDA_LIB`

## Build the model assets

1. Export ONNX wrappers:

```bash
uv run \
  --with 'huggingface_hub>=0.33.0' \
  --with 'onnx>=1.17.0' \
  --with 'onnxscript>=0.1.0' \
  --with 'sentencepiece>=0.2.0' \
  --with 'torch>=2.6.0' \
  --with 'transformers>=4.53.0' \
  python scripts/semantic-safety/export_qwen_onnx.py --output-dir /tmp/semantic-safety-onnx
```

2. Build TensorRT engines:

```bash
scripts/semantic-safety/build_trt_engines.sh /tmp/semantic-safety-onnx /opt/semantic-safety/engines
```

The engine builder defaults to `fp32` for correctness. Set
`SEMANTIC_SAFETY_TRT_PRECISION=fp16` only after validating output quality on
your target GPU.

3. Validate the final assets:

```bash
scripts/semantic-safety/validate_assets.sh /opt/semantic-safety/engines /tmp/semantic-safety-onnx/tokenizer
```

Copy the tokenizer directory to the runtime path you plan to use for
`SEMANTIC_SAFETY_TOKENIZER_DIR`.

## Service runtime

Start from [semantic-safety-service.env.example](/Users/krystian/code/tiny-reverse-proxy-rust/configs/semantic-safety-service.env.example).

Important runtime expectations:

- `SEMANTIC_SAFETY_EMBEDDING_ENGINE`, `SEMANTIC_SAFETY_RERANKER_ENGINE`, and
  `SEMANTIC_SAFETY_TOKENIZER_DIR` are required.
- The service binary must be compiled with `--features native-tensorrt`; a
  default build will fail fast at startup instead of silently using fake
  inference.
- Startup fails fast if assets are missing or warmup fails.
- `SEMANTIC_SAFETY_METRICS_ADDR` enables a Prometheus `/metrics` endpoint.
- The service health RPC returns `ready=true` only after both engines load and
  warmup succeeds.

## Gateway runtime

Use [semantic-safety-gateway.example.toml](/Users/krystian/code/tiny-reverse-proxy-rust/configs/semantic-safety-gateway.example.toml)
as the starting plugin config.

Status expectations:

- `/api/v1/projects/{project_id}/semantic-safety/status` shows project sync
  state plus service readiness/backend message.
- Semantic findings stay observe-only in v0.
- Service failures or degraded responses never block inference; the gateway
  records audit metadata and continues traffic.

## Manual GPU smoke tests

With a native build plus real engine/tokenizer assets in the runtime env vars,
you can run the ignored TensorRT smoke tests:

```bash
cargo test -p semantic-safety-service --features native-tensorrt --test native_backend_smoke -- --ignored --nocapture --test-threads=1
```
