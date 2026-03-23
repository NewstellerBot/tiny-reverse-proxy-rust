#!/usr/bin/env python3
"""Export Qwen semantic safety wrappers to ONNX.

This script emits ONNX graphs that already produce:
- normalized embedding vectors for Qwen3-Embedding-0.6B
- yes-probability rerank scores for Qwen3-Reranker-0.6B

The Rust service then only needs to tokenize inputs and execute the resulting
TensorRT engines in-process.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.export import Dim
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class EmbeddingExportWrapper(torch.nn.Module):
    def __init__(self, model: AutoModel):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        # The service tokenizer left-pads every batch, so the final non-pad token
        # is always at the last sequence position.
        return hidden[:, -1, :]


class RerankerExportWrapper(torch.nn.Module):
    def __init__(self, model: AutoModelForCausalLM, yes_token_id: int, no_token_id: int):
        super().__init__()
        self.model = model.eval()
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Reranker prompts are also left-padded, so next-token logits are taken
        # from the final sequence position.
        final_logits = logits[:, -1, :]
        yes_no = final_logits[:, [self.no_token_id, self.yes_token_id]]
        probs = torch.softmax(yes_no.float(), dim=-1)
        return probs[:, 1]


def export_model(module: torch.nn.Module, inputs: dict[str, torch.Tensor], output_path: Path, output_name: str) -> None:
    dynamic_shapes = (
        {0: Dim("batch"), 1: Dim("sequence")},
        {0: Dim("batch"), 1: Dim("sequence")},
    )
    torch.onnx.export(
        module.eval(),
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=[output_name],
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model-id", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--reranker-model-id", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--tokenizer-id", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    embedding_model = AutoModel.from_pretrained(args.embedding_model_id)
    reranker_model = AutoModelForCausalLM.from_pretrained(args.reranker_model_id)

    sample_inputs = tokenizer(
        [
            "Instruct: classify semantic policy topic\nQuery: Company X layoffs next week",
            "Instruct: classify semantic policy topic\nQuery: Vendor Y acquisition rumor and board discussion",
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    yes_token_id = tokenizer.convert_tokens_to_ids("yes")
    no_token_id = tokenizer.convert_tokens_to_ids("no")
    if yes_token_id is None or no_token_id is None:
        raise RuntimeError("tokenizer must expose yes/no token ids for reranker export")

    export_model(
        EmbeddingExportWrapper(embedding_model),
        sample_inputs,
        output_dir / "embedding.onnx",
        "embeddings",
    )
    export_model(
        RerankerExportWrapper(reranker_model, yes_token_id, no_token_id),
        sample_inputs,
        output_dir / "reranker.onnx",
        "scores",
    )

    print(f"Exported ONNX artifacts to {output_dir}")


if __name__ == "__main__":
    main()
