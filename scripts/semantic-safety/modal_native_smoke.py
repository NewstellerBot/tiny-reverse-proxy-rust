#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

import modal


APP_NAME = "semantic-safety-native-smoke"
DEFAULT_MODAL_IMAGE = os.environ.get(
    "SEMANTIC_SAFETY_MODAL_IMAGE", "nvcr.io/nvidia/tensorrt:24.10-py3"
)
EXPORT_UV_DEPS = [
    "huggingface_hub>=0.33.0",
    "onnx>=1.17.0",
    "onnxscript>=0.1.0",
    "sentencepiece>=0.2.0",
    "torch>=2.6.0",
    "transformers>=4.53.0",
]
REMOTE_REPO = "/root/repo"
ONNX_DIR = "/tmp/semantic-safety-onnx"
ENGINE_DIR = "/tmp/semantic-safety-engines"
TOKENIZER_DIR = f"{ONNX_DIR}/tokenizer"
DEFAULT_MAX_LENGTH = 256
SMOKE_WORKSPACE_MANIFEST = """[workspace]
members = [
    "crates/semantic-safety-protocol",
    "crates/semantic-safety-trt-sys",
    "crates/semantic-safety-trt",
    "crates/semantic-safety-service",
]
resolver = "2"

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
"""


base_image = modal.Image.from_registry(DEFAULT_MODAL_IMAGE).env(
    {"CARGO_TERM_COLOR": "always"}
)

image = base_image

app = modal.App(APP_NAME)


def run_checked(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str = REMOTE_REPO,
) -> str:
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return completed.stdout


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("no candidate paths existed")


def first_recursive_match(roots: Iterable[Path], pattern: str) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for match in root.rglob(pattern):
            return match
    return None


def resolve_tensorrt_cuda_dirs() -> dict[str, str]:
    python_package_roots = list(Path("/usr/local/lib").glob("python*/site-packages")) + list(
        Path("/usr/local/lib").glob("python*/dist-packages")
    )
    tensorrt_include = (
        first_recursive_match(
            [
                Path("/workspace/tensorrt/include"),
                Path("/usr/include/x86_64-linux-gnu"),
                Path("/usr/include"),
                Path("/usr/local/tensorrt/include"),
                Path("/opt/tensorrt/include"),
                *python_package_roots,
            ],
            "NvInfer.h",
        )
        or first_recursive_match(
            [
                Path("/workspace/tensorrt"),
                Path("/usr"),
                Path("/usr/local"),
                Path("/opt"),
                *python_package_roots,
            ],
            "NvInfer.h",
        )
    )
    if tensorrt_include is None:
        raise FileNotFoundError("could not locate NvInfer.h")
    tensorrt_include = tensorrt_include.parent
    cuda_include = first_existing(
        [
            Path("/usr/local/cuda/include/cuda_runtime_api.h"),
            Path("/usr/include/cuda_runtime_api.h"),
            Path("/usr/include/x86_64-linux-gnu/cuda_runtime_api.h"),
        ]
    ).parent
    tensorrt_lib = (
        first_recursive_match(
            [
                Path("/workspace/tensorrt/lib"),
                Path("/workspace/tensorrt/lib64"),
                Path("/usr/lib/x86_64-linux-gnu"),
                Path("/usr/lib"),
                Path("/usr/local/tensorrt/lib"),
                Path("/usr/local/tensorrt/lib64"),
                Path("/opt/tensorrt/lib"),
                Path("/opt/tensorrt/lib64"),
                *python_package_roots,
            ],
            "libnvinfer.so*",
        )
        or first_recursive_match(
            [
                Path("/workspace/tensorrt"),
                Path("/usr"),
                Path("/usr/local"),
                Path("/opt"),
                *python_package_roots,
            ],
            "libnvinfer.so*",
        )
    )
    if tensorrt_lib is None:
        raise FileNotFoundError("could not locate libnvinfer.so")
    tensorrt_lib = tensorrt_lib.parent
    cuda_lib = first_existing(
        [
            Path("/usr/local/cuda/lib64/libcudart.so"),
            Path("/usr/lib/x86_64-linux-gnu/libcudart.so"),
            Path("/usr/lib64/libcudart.so"),
            Path("/usr/lib/libcudart.so"),
        ]
    ).parent

    return {
        "SEMANTIC_SAFETY_TENSORRT_INCLUDE": str(tensorrt_include),
        "SEMANTIC_SAFETY_TENSORRT_LIB": str(tensorrt_lib),
        "SEMANTIC_SAFETY_CUDA_INCLUDE": str(cuda_include),
        "SEMANTIC_SAFETY_CUDA_LIB": str(cuda_lib),
    }


def locate_trtexec(env: dict[str, str]) -> str | None:
    candidates = [
        shutil.which("trtexec", path=env.get("PATH")),
        "/workspace/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def bootstrap_runtime_env(base_env: dict[str, str]) -> dict[str, str]:
    run_checked(
        [
            "bash",
            "-lc",
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "build-essential ca-certificates curl git pkg-config protobuf-compiler",
        ],
        env=base_env,
        cwd="/",
    )
    run_checked(
        [
            "bash",
            "-lc",
            "command -v uv >/dev/null 2>&1 || "
            "(curl -LsSf https://astral.sh/uv/install.sh | sh)",
        ],
        env=base_env,
        cwd="/",
    )
    run_checked(
        [
            "bash",
            "-lc",
            "command -v cargo >/dev/null 2>&1 || "
            "(curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal)",
        ],
        env=base_env,
        cwd="/",
    )

    enriched_env = base_env.copy()
    enriched_env["PATH"] = (
        f"/root/.local/bin:/root/.cargo/bin:{enriched_env.get('PATH', '')}"
    )
    enriched_env["UV_CACHE_DIR"] = "/root/.cache/uv"
    return enriched_env


def build_repo_archive() -> bytes:
    repo_root = Path(__file__).resolve().parents[2]
    include_paths = [
        repo_root / "Cargo.toml",
        repo_root / "Cargo.lock",
        repo_root / "crates/semantic-safety-service",
        repo_root / "crates/semantic-safety-protocol",
        repo_root / "crates/semantic-safety-trt",
        repo_root / "crates/semantic-safety-trt-sys",
        repo_root / "scripts/semantic-safety",
    ]

    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in include_paths:
            tar.add(path, arcname=path.relative_to(repo_root))
    return buffer.getvalue()


def extract_repo_archive(repo_archive: bytes) -> None:
    if os.path.exists(REMOTE_REPO):
        shutil.rmtree(REMOTE_REPO)
    os.makedirs(REMOTE_REPO, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(repo_archive), mode="r:gz") as tar:
        tar.extractall(REMOTE_REPO)
    (Path(REMOTE_REPO) / "Cargo.toml").write_text(SMOKE_WORKSPACE_MANIFEST)


@app.function(gpu="L4", cpu=8, memory=32768, timeout=7200, image=image)
def run_native_smoke(
    repo_archive: bytes,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, str]:
    extract_repo_archive(repo_archive)

    if os.path.exists(ONNX_DIR):
        shutil.rmtree(ONNX_DIR)
    if os.path.exists(ENGINE_DIR):
        shutil.rmtree(ENGINE_DIR)

    build_env = bootstrap_runtime_env(os.environ.copy())

    run_checked(["uv", "--version"], env=build_env)
    run_checked(["uv", "run", "python", "--version"], env=build_env)
    run_checked(["cargo", "--version"], env=build_env)
    run_checked(["rustc", "--version"], env=build_env)
    run_checked(["nvidia-smi"], env=build_env)
    trtexec_path = locate_trtexec(build_env)
    if not trtexec_path:
        find_output = run_checked(
            ["bash", "-lc", "find / -name trtexec 2>/dev/null | head -20"],
            env=build_env,
            cwd="/",
        ).strip()
        if find_output:
            first_match = find_output.splitlines()[0].strip()
            if first_match:
                trtexec_path = first_match

    if trtexec_path:
        build_env["TRTEXEC_BIN"] = trtexec_path
        build_env["PATH"] = (
            f"{Path(trtexec_path).parent}:{build_env.get('PATH', '')}"
        )
        run_checked(
            ["bash", "-lc", f"'{trtexec_path}' --help >/dev/null"],
            env=build_env,
        )
    else:
        run_checked(
            [
                "uv",
                "run",
                "--with",
                "tensorrt-cu13",
                "python",
                "-c",
                "import tensorrt as trt; print(trt.__version__)",
            ],
            env=build_env,
        )

    build_env.update(resolve_tensorrt_cuda_dirs())
    build_env.update(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "PYTHONUNBUFFERED": "1",
            "RUST_BACKTRACE": "1",
            "SEMANTIC_SAFETY_EMBEDDING_ENGINE": f"{ENGINE_DIR}/embedding.engine",
            "SEMANTIC_SAFETY_RERANKER_ENGINE": f"{ENGINE_DIR}/reranker.engine",
            "SEMANTIC_SAFETY_TOKENIZER_DIR": TOKENIZER_DIR,
            "SEMANTIC_SAFETY_DEVICE_ID": "0",
            "SEMANTIC_SAFETY_MAX_BATCH_SIZE": "8",
            "SEMANTIC_SAFETY_MAX_SEQUENCE_LENGTH": str(max_length),
            "SEMANTIC_SAFETY_TRT_PRECISION": "fp32",
            "SEMANTIC_SAFETY_WARMUP_ENABLED": "true",
        }
    )

    export_command = ["uv", "run"]
    for dep in EXPORT_UV_DEPS:
        export_command.extend(["--with", dep])
    export_command.extend(
        [
            "python",
            "scripts/semantic-safety/export_qwen_onnx.py",
            "--output-dir",
            ONNX_DIR,
            "--max-length",
            str(max_length),
        ]
    )
    run_checked(export_command, env=build_env)
    run_checked(
        [
            "bash",
            "scripts/semantic-safety/build_trt_engines.sh",
            ONNX_DIR,
            ENGINE_DIR,
        ],
        env=build_env,
    )
    run_checked(
        [
            "bash",
            "scripts/semantic-safety/validate_assets.sh",
            ENGINE_DIR,
            TOKENIZER_DIR,
        ],
        env=build_env,
    )
    test_output = run_checked(
        [
            "cargo",
            "test",
            "-p",
            "semantic-safety-service",
            "--features",
            "native-tensorrt",
            "--test",
            "native_backend_smoke",
            "--",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ],
        env=build_env,
    )
    return {
        "engine_dir": ENGINE_DIR,
        "tokenizer_dir": TOKENIZER_DIR,
        "max_length": str(max_length),
        "test_output_tail": "\n".join(test_output.strip().splitlines()[-20:]),
    }


@app.local_entrypoint()
def main(max_length: int = DEFAULT_MAX_LENGTH) -> None:
    repo_archive = build_repo_archive()
    result = run_native_smoke.remote(repo_archive=repo_archive, max_length=max_length)
    print(result)
