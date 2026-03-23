#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <onnx-dir> <engine-dir>" >&2
  exit 1
fi

ONNX_DIR="$1"
ENGINE_DIR="$2"
mkdir -p "$ENGINE_DIR"

EMBEDDING_ONNX="$ONNX_DIR/embedding.onnx"
RERANKER_ONNX="$ONNX_DIR/reranker.onnx"

if [[ ! -f "$EMBEDDING_ONNX" || ! -f "$RERANKER_ONNX" ]]; then
  echo "missing ONNX exports in $ONNX_DIR" >&2
  exit 1
fi

TRTEXEC_BIN="${TRTEXEC_BIN:-}"
TRT_PRECISION="${SEMANTIC_SAFETY_TRT_PRECISION:-fp32}"
if [[ -z "$TRTEXEC_BIN" ]]; then
  for candidate in \
    "$(command -v trtexec 2>/dev/null || true)" \
    /workspace/tensorrt/bin/trtexec \
    /usr/src/tensorrt/bin/trtexec \
    /usr/local/tensorrt/bin/trtexec \
    /opt/tensorrt/bin/trtexec \
    /usr/local/bin/trtexec \
    /usr/bin/trtexec
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      TRTEXEC_BIN="$candidate"
      break
    fi
  done
fi

TRT_PRECISION_FLAG=()
case "$TRT_PRECISION" in
  fp32)
    ;;
  fp16)
    TRT_PRECISION_FLAG+=(--fp16)
    ;;
  *)
    echo "unsupported SEMANTIC_SAFETY_TRT_PRECISION: $TRT_PRECISION" >&2
    exit 1
    ;;
esac

if [[ -n "$TRTEXEC_BIN" ]]; then
  "$TRTEXEC_BIN" \
    --onnx="$EMBEDDING_ONNX" \
    --saveEngine="$ENGINE_DIR/embedding.engine" \
    --minShapes=input_ids:1x32,attention_mask:1x32 \
    --optShapes=input_ids:4x256,attention_mask:4x256 \
    --maxShapes=input_ids:8x512,attention_mask:8x512 \
    "${TRT_PRECISION_FLAG[@]}"

  "$TRTEXEC_BIN" \
    --onnx="$RERANKER_ONNX" \
    --saveEngine="$ENGINE_DIR/reranker.engine" \
    --minShapes=input_ids:1x32,attention_mask:1x32 \
    --optShapes=input_ids:4x256,attention_mask:4x256 \
    --maxShapes=input_ids:8x512,attention_mask:8x512 \
    "${TRT_PRECISION_FLAG[@]}"
else
  uv run --with tensorrt-cu13 python scripts/semantic-safety/build_trt_engines.py \
    --precision "$TRT_PRECISION" \
    "$ONNX_DIR" \
    "$ENGINE_DIR"
fi

echo "Built TensorRT engines in $ENGINE_DIR"
