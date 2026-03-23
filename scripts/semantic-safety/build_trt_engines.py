#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("onnx_dir")
    parser.add_argument("engine_dir")
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--opt-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--min-sequence-length", type=int, default=32)
    parser.add_argument("--opt-sequence-length", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--workspace-bytes", type=int, default=4 * 1024 * 1024 * 1024)
    return parser.parse_args()


def build_engine(
    logger: trt.Logger,
    onnx_path: Path,
    engine_path: Path,
    *,
    min_batch_size: int,
    opt_batch_size: int,
    max_batch_size: int,
    min_sequence_length: int,
    opt_sequence_length: int,
    max_sequence_length: int,
    workspace_bytes: int,
    precision: str,
) -> None:
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open("rb") as handle:
        if not parser.parse(handle.read()):
            errors = "\n".join(
                str(parser.get_error(idx)) for idx in range(parser.num_errors)
            )
            raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("TensorRT platform does not support fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    min_shape = (min_batch_size, min_sequence_length)
    opt_shape = (opt_batch_size, opt_sequence_length)
    max_shape = (max_batch_size, max_sequence_length)
    profile.set_shape("input_ids", min_shape, opt_shape, max_shape)
    profile.set_shape("attention_mask", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build an engine for {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with engine_path.open("wb") as handle:
        handle.write(serialized)


def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)

    logger = trt.Logger(trt.Logger.WARNING)
    build_engine(
        logger,
        onnx_dir / "embedding.onnx",
        engine_dir / "embedding.engine",
        min_batch_size=args.min_batch_size,
        opt_batch_size=args.opt_batch_size,
        max_batch_size=args.max_batch_size,
        min_sequence_length=args.min_sequence_length,
        opt_sequence_length=args.opt_sequence_length,
        max_sequence_length=args.max_sequence_length,
        workspace_bytes=args.workspace_bytes,
        precision=args.precision,
    )
    build_engine(
        logger,
        onnx_dir / "reranker.onnx",
        engine_dir / "reranker.engine",
        min_batch_size=args.min_batch_size,
        opt_batch_size=args.opt_batch_size,
        max_batch_size=args.max_batch_size,
        min_sequence_length=args.min_sequence_length,
        opt_sequence_length=args.opt_sequence_length,
        max_sequence_length=args.max_sequence_length,
        workspace_bytes=args.workspace_bytes,
        precision=args.precision,
    )
    print(f"Built TensorRT engines in {engine_dir}")


if __name__ == "__main__":
    main()
