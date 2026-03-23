#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <engine-dir> <tokenizer-dir>" >&2
  exit 1
fi

ENGINE_DIR="$1"
TOKENIZER_DIR="$2"

for required in "$ENGINE_DIR/embedding.engine" "$ENGINE_DIR/reranker.engine" "$TOKENIZER_DIR/tokenizer.json"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required asset: $required" >&2
    exit 1
  fi
done

echo "Semantic safety assets look present:"
echo "  embedding.engine"
echo "  reranker.engine"
echo "  tokenizer.json"
