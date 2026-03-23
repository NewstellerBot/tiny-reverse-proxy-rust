#!/usr/bin/env python3
"""
Run a reproducible OSS gateway benchmark under identical Docker resource caps.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "bench" / "oss-gateway-shootout"
RESULTS_DIR = BENCH_DIR / "results"

NETWORK_NAME = "oss-gateway-shootout"
MOCK_CONTAINER_IMAGE = "python:3.13-slim"
LITELLM_IMAGE = "docker.litellm.ai/berriai/litellm:main-latest"
BIFROST_IMAGE = "maximhq/bifrost:latest"
TINY_PROXY_IMAGE = "tiny-reverse-proxy-bench:local"
MAX_RAW_OUTPUT_CHARS = 8000

MOCK_CONTAINER_PORT = 9900
MOCK_HOST_PORT = 19900
MOCK_GATEWAY_HOST = "host.docker.internal"
TINY_PROXY_CONTAINER_PORT = 8080
TINY_PROXY_MANAGEMENT_CONTAINER_PORT = 9090
LITELLM_CONTAINER_PORT = 4000
BIFROST_CONTAINER_PORT = 8080
BENCHMARK_MANAGEMENT_TOKEN = "bench-admin-token"
BENCHMARK_PROJECT_PREFIX = "oss-bench"

DEFAULT_TARGETS = ["direct", "tiny-proxy", "litellm", "bifrost"]
DEFAULT_SCENARIO = "pass-through"
SCENARIO_CHOICES = [
    DEFAULT_SCENARIO,
    "streaming",
    "prompt-cache",
    "tool-round-trip",
    "prompt-cache-affinity",
    "prompt-cache-affinity-routing-only",
    "semantic-cache-affinity",
]
MANAGED_BOOTSTRAP_SCENARIOS = {
    "prompt-cache",
    "tool-round-trip",
    "prompt-cache-affinity",
    "prompt-cache-affinity-routing-only",
    "semantic-cache-affinity",
}

PROMPT_CACHE_AFFINITY_SCENARIOS = {
    "prompt-cache-affinity",
    "prompt-cache-affinity-routing-only",
}


@dataclass
class RequestVariant:
    path: str
    body: dict[str, Any]
    headers: dict[str, str]


@dataclass
class TargetSpec:
    name: str
    image: str | None
    container_port: int | None
    host_port: int | None
    request_variants: list[RequestVariant]


def run_command(
    args: list[str],
    *,
    cwd: pathlib.Path | None = None,
    check: bool = True,
    capture_output: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=True,
        env=env,
    )


def truncate_text(value: str, limit: int = MAX_RAW_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    head = limit // 2
    tail = limit - head
    return value[:head] + "\n...\n" + value[-tail:]


def require_command(name: str) -> None:
    result = run_command(["/usr/bin/env", "sh", "-lc", f"command -v {shlex.quote(name)}"], check=False)
    if result.returncode != 0:
        raise SystemExit(f"required command is missing: {name}")


def ensure_docker_available() -> None:
    result = run_command(["docker", "info"], check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown docker error"
        raise SystemExit(f"Docker daemon is not available: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OSS AI gateways with identical Docker caps.")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=DEFAULT_TARGETS,
        default=DEFAULT_TARGETS,
        help="Which targets to benchmark.",
    )
    parser.add_argument("--cpus", default="1.0", help="Docker CPU limit for every gateway container.")
    parser.add_argument("--memory", default="512m", help="Docker memory limit for every gateway container.")
    parser.add_argument("--duration", default="15s", help="hey benchmark duration.")
    parser.add_argument("--concurrency", type=int, default=32, help="hey worker concurrency.")
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_CHOICES,
        default=DEFAULT_SCENARIO,
        help="Benchmark scenario to run.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Directory where benchmark artifacts are written.",
    )
    parser.add_argument(
        "--rebuild-tiny-image",
        action="store_true",
        help="Force rebuilding the tiny-reverse-proxy benchmark image.",
    )
    parser.add_argument(
        "--pull-images",
        action="store_true",
        help="Pull competitor images before the run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render configs and planned commands without invoking Docker or hey.",
    )
    parser.add_argument(
        "--gateway-log-driver",
        default="none",
        help="Docker log driver for gateway containers. Use json-file if you need docker logs for debugging.",
    )
    return parser.parse_args()


def benchmark_targets(scenario: str) -> dict[str, TargetSpec]:
    short_prompt = [{"role": "user", "content": "reply with pong"}]
    request_body: dict[str, Any] = {"model": "benchmark-model", "messages": short_prompt}
    if scenario == "streaming":
        request_body["stream"] = True
    tiny_proxy_body = request_body.copy()
    direct_body = request_body.copy()
    litellm_body = request_body.copy()
    bifrost_body = request_body.copy()
    if scenario == "prompt-cache":
        tiny_proxy_body["trp_prompt_cache"] = {
            "enabled": True,
            "ttl": "24h",
            "key": "tenant:bench",
        }
        for body in (direct_body, litellm_body, bifrost_body):
            body["prompt_cache_key"] = "tenant:bench"
            body["prompt_cache_retention"] = "24h"
    elif scenario in PROMPT_CACHE_AFFINITY_SCENARIOS:
        tiny_proxy_body = {
            "model": "benchmark-model",
            "messages": [{"role": "user", "content": "use the warm cache again"}],
            "trp_prompt_cache": {
                "enabled": True,
                "ttl": "24h",
                "key": "tenant:bench-affinity",
            },
        }
        direct_body = {
            "model": "benchmark-model",
            "messages": [{"role": "user", "content": "use the warm cache again"}],
            "prompt_cache_key": "tenant:bench-affinity",
            "prompt_cache_retention": "24h",
        }
    elif scenario == "semantic-cache-affinity":
        tiny_proxy_body = {
            "model": "benchmark-model",
            "messages": [{"role": "user", "content": "need password reset help"}],
            "trp_semantic_cache": {
                "enabled": True,
                "ttl_secs": 600,
                "similarity_threshold": 0.7,
            },
        }
        direct_body = {
            "model": "benchmark-model",
            "messages": [{"role": "user", "content": "need password reset help"}],
        }
    elif scenario == "tool-round-trip":
        tiny_proxy_body["trp_tools"] = {
            "enabled": True,
            "names": ["tool_echo"],
        }
        direct_body["benchmark_direct_tool_round_trip"] = True
    targets = {
        "direct": TargetSpec(
            name="direct",
            image=None,
            container_port=None,
            host_port=MOCK_HOST_PORT,
            request_variants=[
                RequestVariant(
                    path={
                        "prompt-cache-affinity": "/alpha/v1/chat/completions",
                        "prompt-cache-affinity-routing-only": "/alpha/v1/chat/completions",
                        "semantic-cache-affinity": "/semantic-cache-hit/v1/chat/completions",
                    }.get(scenario, "/v1/chat/completions"),
                    body=direct_body,
                    headers={"Content-Type": "application/json"},
                )
            ],
        ),
        "tiny-proxy": TargetSpec(
            name="tiny-proxy",
            image=TINY_PROXY_IMAGE,
            container_port=TINY_PROXY_CONTAINER_PORT,
            host_port=18080,
            request_variants=[
                RequestVariant(
                    path="/v1/chat/completions",
                    body=tiny_proxy_body,
                    headers={"Content-Type": "application/json"},
                )
            ],
        ),
        "litellm": TargetSpec(
            name="litellm",
            image=LITELLM_IMAGE,
            container_port=LITELLM_CONTAINER_PORT,
            host_port=18081,
            request_variants=[
                RequestVariant(
                    path="/v1/chat/completions",
                    body=litellm_body,
                    headers={"Content-Type": "application/json"},
                ),
                RequestVariant(
                    path="/chat/completions",
                    body=litellm_body,
                    headers={"Content-Type": "application/json"},
                ),
            ],
        ),
        "bifrost": TargetSpec(
            name="bifrost",
            image=BIFROST_IMAGE,
            container_port=BIFROST_CONTAINER_PORT,
            host_port=18082,
            request_variants=[
                RequestVariant(
                    path="/v1/chat/completions",
                    body={**bifrost_body, "model": "openai/benchmark-model"},
                    headers={"Content-Type": "application/json"},
                ),
                RequestVariant(
                    path="/chat/completions",
                    body={**bifrost_body, "model": "openai/benchmark-model"},
                    headers={"Content-Type": "application/json"},
                ),
            ],
        ),
    }
    if scenario in PROMPT_CACHE_AFFINITY_SCENARIOS | {"semantic-cache-affinity"}:
        return {name: spec for name, spec in targets.items() if name in {"direct", "tiny-proxy"}}
    return targets


def make_tiny_proxy_config() -> str:
    return make_tiny_proxy_config_for_scenario(DEFAULT_SCENARIO)


def mock_gateway_upstream(path: str = "") -> str:
    suffix = path if not path or path.startswith("/") else f"/{path}"
    return f"http://{MOCK_GATEWAY_HOST}:{MOCK_HOST_PORT}{suffix}"


def make_tiny_proxy_config_for_scenario(scenario: str) -> str:
    def canonical_openai_provider_block(
        name: str,
        base_url: str,
        *,
        prompt_cache: bool = False,
        tools: bool = False,
    ) -> str:
        lines = [
            "[[providers]]",
            f'name = "{name}"',
            'api_key = "sk-benchmark-upstream"',
            f'base_url = "{base_url}"',
            'models = ["benchmark-model"]',
            'family = "openai"',
            "",
            "[providers.surfaces]",
            'responses = "openai_compatible"',
            'files = "openai_compatible"',
            'batches = "openai_compatible"',
        ]
        if tools:
            lines.append('tools = "openai"')
        if prompt_cache:
            lines.extend(
                [
                    "",
                    "[providers.surfaces.prompt_cache]",
                    'protocol = "openai"',
                    'request_controls = true',
                ]
            )
        return "\n".join(lines)

    if scenario in PROMPT_CACHE_AFFINITY_SCENARIOS:
        return textwrap.dedent(
            f"""
            port = {TINY_PROXY_CONTAINER_PORT}
            management_api_port = {TINY_PROXY_MANAGEMENT_CONTAINER_PORT}
            management_api_token = "{BENCHMARK_MANAGEMENT_TOKEN}"
            store_url = "sqlite:///tmp/oss-gateway-shootout-{scenario}.db"

            [paths]
            "/**" = ["{MOCK_GATEWAY_HOST}:{MOCK_HOST_PORT}"]

            {canonical_openai_provider_block(
                "alpha",
                mock_gateway_upstream('/alpha'),
                prompt_cache=True,
            )}

            {canonical_openai_provider_block(
                "beta",
                mock_gateway_upstream('/beta'),
                prompt_cache=True,
            )}

            [[plugins]]
            name = "prompt_cache"
            enabled = true

            [plugins.config]
            anthropic_default_scope = "auto"
            persist_routing_hints = {"false" if scenario == "prompt-cache-affinity-routing-only" else "true"}
            """
        ).strip() + "\n"
    if scenario == "semantic-cache-affinity":
        return textwrap.dedent(
            f"""
            port = {TINY_PROXY_CONTAINER_PORT}
            management_api_port = {TINY_PROXY_MANAGEMENT_CONTAINER_PORT}
            management_api_token = "{BENCHMARK_MANAGEMENT_TOKEN}"
            store_url = "sqlite:///tmp/oss-gateway-shootout-semantic-cache-affinity.db"

            [paths]
            "/**" = ["{MOCK_GATEWAY_HOST}:{MOCK_HOST_PORT}"]

            {canonical_openai_provider_block("alpha", mock_gateway_upstream('/alpha'))}

            {canonical_openai_provider_block("beta", mock_gateway_upstream('/beta'))}

            [[plugins]]
            name = "semantic_cache"
            enabled = true

            [plugins.config]
            default_ttl_secs = 600
            default_similarity_threshold = 0.8
            max_entries = 32
            """
        ).strip() + "\n"
    if scenario == "prompt-cache":
        return textwrap.dedent(
            f"""
            port = {TINY_PROXY_CONTAINER_PORT}
            management_api_port = {TINY_PROXY_MANAGEMENT_CONTAINER_PORT}
            management_api_token = "{BENCHMARK_MANAGEMENT_TOKEN}"
            store_url = "sqlite:///tmp/oss-gateway-shootout-prompt-cache.db"

            [paths]
            "/**" = ["{MOCK_GATEWAY_HOST}:{MOCK_HOST_PORT}"]

            {canonical_openai_provider_block(
                "openai",
                mock_gateway_upstream(),
                prompt_cache=True,
            )}

            [[plugins]]
            name = "prompt_cache"
            enabled = true

            [plugins.config]
            anthropic_default_scope = "auto"
            """
        ).strip() + "\n"
    if scenario == "tool-round-trip":
        return textwrap.dedent(
            f"""
            port = {TINY_PROXY_CONTAINER_PORT}
            management_api_port = {TINY_PROXY_MANAGEMENT_CONTAINER_PORT}
            management_api_token = "{BENCHMARK_MANAGEMENT_TOKEN}"
            store_url = "sqlite:///tmp/oss-gateway-shootout-tool-runtime.db"

            [paths]
            "/**" = ["{MOCK_GATEWAY_HOST}:{MOCK_HOST_PORT}"]

            {canonical_openai_provider_block(
                "openai",
                mock_gateway_upstream(),
                tools=True,
            )}

            [[plugins]]
            name = "tool_runtime"
            enabled = true

            [plugins.config]
            tool_timeout_ms = 5000
            max_round_trips = 4
            """
        ).strip() + "\n"
    return textwrap.dedent(
        f"""
        port = {TINY_PROXY_CONTAINER_PORT}

        [paths]
        "/**" = ["{MOCK_GATEWAY_HOST}:{MOCK_HOST_PORT}"]
        """
    ).strip() + "\n"


def make_litellm_config() -> str:
    return textwrap.dedent(
        """
        model_list:
          - model_name: benchmark-model
            litellm_params:
              model: openai/benchmark-model
              api_base: http://host.docker.internal:19900/v1
              api_key: sk-benchmark-upstream
        """
    ).strip() + "\n"


def make_bifrost_config() -> str:
    return json.dumps(
        {
            "providers": {
                "openai": {
                    "keys": [
                        {
                            "name": "benchmark-key",
                            "value": "sk-benchmark-upstream",
                            "models": ["benchmark-model"],
                            "weight": 1.0,
                        }
                    ],
                    "network_config": {
                        "base_url": mock_gateway_upstream("/v1"),
                        "max_retries": 0,
                    },
                }
            }
        },
        indent=2,
    ) + "\n"


def make_litellm_log_config() -> str:
    return textwrap.dedent(
        """
        version: 1
        disable_existing_loggers: false
        formatters:
          default:
            format: '%(levelprefix)s %(message)s'
        handlers:
          default:
            class: logging.StreamHandler
            formatter: default
            stream: ext://sys.stderr
        loggers:
          uvicorn:
            level: WARNING
            handlers: [default]
            propagate: false
          uvicorn.error:
            level: WARNING
            handlers: [default]
            propagate: false
          uvicorn.access:
            level: CRITICAL
            handlers: [default]
            propagate: false
          litellm:
            level: WARNING
            handlers: [default]
            propagate: false
        root:
          level: WARNING
          handlers: [default]
        """
    ).strip() + "\n"


def parse_duration_to_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m)", value)
    if not match:
        raise ValueError(f"unsupported duration format: {value}")
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "ms":
        return number / 1000.0
    if unit == "s":
        return number
    return number * 60.0


def parse_byte_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?i?B|B)", value.strip())
    if not match:
        raise ValueError(f"unsupported byte size: {value}")
    number = float(match.group(1))
    unit = match.group(2)
    multipliers = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
        "PiB": 1024**5,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "PB": 1000**5,
    }
    return int(number * multipliers[unit])


def ensure_results_dir(base_dir: pathlib.Path) -> pathlib.Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = base_dir / timestamp
    suffix = 1
    while path.exists():
        path = base_dir / f"{timestamp}-{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    (path / "configs").mkdir()
    return path


def write_configs(results_dir: pathlib.Path, scenario: str) -> dict[str, pathlib.Path]:
    bifrost_dir = results_dir / "configs" / "bifrost"
    bifrost_dir.mkdir()
    configs = {
        "tiny-proxy": results_dir / "configs" / "tiny-proxy.toml",
        "litellm": results_dir / "configs" / "litellm-config.yaml",
        "litellm-log-config": results_dir / "configs" / "litellm-logging.yaml",
        "bifrost": bifrost_dir / "config.json",
    }
    configs["tiny-proxy"].write_text(make_tiny_proxy_config_for_scenario(scenario))
    configs["litellm"].write_text(make_litellm_config())
    configs["litellm-log-config"].write_text(make_litellm_log_config())
    configs["bifrost"].write_text(make_bifrost_config())
    return configs


def create_network(name: str) -> None:
    run_command(["docker", "network", "create", name], check=False)


def remove_network(name: str) -> None:
    run_command(["docker", "network", "rm", "-f", name], check=False)


def stop_container(name: str) -> None:
    run_command(["docker", "rm", "-f", name], check=False)


def maybe_pull_images(args: argparse.Namespace, specs: dict[str, TargetSpec]) -> None:
    if not args.pull_images:
        return
    pulled = set()
    for target_name in args.targets:
        spec = specs[target_name]
        if spec.image is None or spec.image in pulled or spec.image == TINY_PROXY_IMAGE:
            continue
        print(f"pulling {spec.image}")
        run_command(["docker", "pull", spec.image])
        pulled.add(spec.image)
    if MOCK_CONTAINER_IMAGE not in pulled:
        print(f"pulling {MOCK_CONTAINER_IMAGE}")
        run_command(["docker", "pull", MOCK_CONTAINER_IMAGE])


def ensure_tiny_proxy_image(rebuild: bool) -> None:
    if not rebuild:
        inspect = run_command(["docker", "image", "inspect", TINY_PROXY_IMAGE], check=False)
        if inspect.returncode == 0:
            return
    run_command(
        [
            "docker",
            "build",
            "-f",
            str(BENCH_DIR / "Dockerfile.tiny-proxy"),
            "-t",
            TINY_PROXY_IMAGE,
            ".",
        ],
        cwd=ROOT,
    )


def start_mock_upstream(network_name: str, host_port: int) -> str:
    container_name = f"{NETWORK_NAME}-mock-{os.getpid()}"
    stop_container(container_name)
    run_command(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "--network",
            network_name,
            "--network-alias",
            "mock-llm",
            "-p",
            f"{host_port}:{MOCK_CONTAINER_PORT}",
            "-v",
            f"{ROOT}:/workspace:ro",
            MOCK_CONTAINER_IMAGE,
            "python",
            "/workspace/bench/mock_llm_api.py",
            "--quiet",
            "--port",
            str(MOCK_CONTAINER_PORT),
        ]
    )
    return container_name


def start_gateway_container(
    target: TargetSpec,
    config_paths: dict[str, pathlib.Path],
    network_name: str,
    cpus: str,
    memory: str,
    log_driver: str,
    scenario: str,
) -> str:
    container_name = f"{NETWORK_NAME}-{target.name}-{os.getpid()}"
    stop_container(container_name)
    common = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "--network",
        network_name,
        "--log-driver",
        log_driver,
        "--cpus",
        cpus,
        "--memory",
        memory,
        "-p",
        f"{target.host_port}:{target.container_port}",
    ]
    if platform.system() == "Linux":
        common.extend(["--add-host", "host.docker.internal:host-gateway"])

    if target.name == "tiny-proxy":
        config_path = config_paths[target.name]
        cmd = common + [
            "-e",
            "RUST_LOG=warn",
            "-v",
            f"{config_path}:/app/config.toml:ro",
        ]
        cmd.extend([target.image, "/app/config.toml"])
    elif target.name == "litellm":
        config_path = config_paths[target.name]
        cmd = common + [
            "-v",
            f"{config_path}:/app/config.yaml:ro",
            "-v",
            f"{config_paths['litellm-log-config']}:/app/logging.yaml:ro",
            target.image,
            "--config",
            "/app/config.yaml",
            "--log_config",
            "/app/logging.yaml",
            "--telemetry",
            "False",
        ]
    elif target.name == "bifrost":
        config_path = config_paths[target.name]
        cmd = common + [
            "-v",
            f"{config_path.parent}:/app/data",
            "-e",
            f"APP_PORT={BIFROST_CONTAINER_PORT}",
            "-e",
            "APP_HOST=0.0.0.0",
            "-e",
            "APP_DIR=/app/data",
            "-e",
            "LOG_LEVEL=error",
            target.image,
        ]
    else:
        raise ValueError(f"unsupported target: {target.name}")

    result = run_command(cmd, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"docker run exited with status {result.returncode}"
        raise RuntimeError(f"failed to start {target.name}: {detail}")
    return container_name


def http_request(host_port: int, variant: RequestVariant, timeout: float = 5.0) -> tuple[int, str]:
    status, _headers, body = http_request_detail(host_port, variant, timeout=timeout)
    return status, body


def http_request_detail(
    host_port: int,
    variant: RequestVariant,
    timeout: float = 5.0,
) -> tuple[int, dict[str, str], str]:
    url = f"http://127.0.0.1:{host_port}{variant.path}"
    data = json.dumps(variant.body).encode()
    req = Request(url, data=data, headers=variant.headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            headers = {key.lower(): value for key, value in resp.headers.items()}
            return resp.status, headers, resp.read().decode()
    except HTTPError as exc:
        headers = {key.lower(): value for key, value in exc.headers.items()}
        return exc.code, headers, exc.read().decode()
    except URLError as exc:
        return 0, {}, str(exc)
    except OSError as exc:
        return 0, {}, str(exc)


def json_request(
    host_port: int,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
) -> tuple[int, str]:
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    url = f"http://127.0.0.1:{host_port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=data, headers=merged_headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except HTTPError as exc:
        return exc.code, exc.read().decode()
    except URLError as exc:
        return 0, str(exc)
    except OSError as exc:
        return 0, str(exc)


def wait_for_host_port(host_port: int, timeout: float = 15.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", host_port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"port {host_port} did not become ready in time")


def choose_working_variant(target: TargetSpec, timeout: float = 20.0) -> RequestVariant:
    wait_for_host_port(target.host_port, timeout=timeout)
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        for variant in target.request_variants:
            status, _body = http_request(target.host_port, variant, timeout=3.0)
            last_status = status
            if status == 200:
                return variant
        time.sleep(0.5)
    raise RuntimeError(f"{target.name} did not return 200 for any request variant (last status={last_status})")


def validate_targets_for_scenario(targets: list[str], scenario: str) -> None:
    supported_targets = {
        "pass-through": set(DEFAULT_TARGETS),
        "streaming": set(DEFAULT_TARGETS),
        "prompt-cache": set(DEFAULT_TARGETS),
        "prompt-cache-affinity-routing-only": {"direct", "tiny-proxy"},
        "prompt-cache-affinity": {"direct", "tiny-proxy"},
        "semantic-cache-affinity": {"direct", "tiny-proxy"},
        "tool-round-trip": {"direct", "tiny-proxy"},
    }[scenario]
    unsupported = sorted(set(targets) - supported_targets)
    if unsupported:
        supported = ", ".join(sorted(supported_targets))
        unsupported_names = ", ".join(unsupported)
        raise SystemExit(
            f"scenario {scenario!r} does not support targets: {unsupported_names}. "
            f"Supported targets: {supported}"
        )


def management_api_request(
    container_name: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[int, str]:
    headers = {
        "Authorization": f"Bearer {BENCHMARK_MANAGEMENT_TOKEN}",
        "Content-Type": "application/json",
    }
    script = textwrap.dedent(
        f"""
        import json
        import os
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        url = "http://127.0.0.1:{TINY_PROXY_MANAGEMENT_CONTAINER_PORT}" + os.environ["TRP_PATH"]
        method = os.environ["TRP_METHOD"]
        headers = json.loads(os.environ["TRP_HEADERS"])
        payload = os.environ.get("TRP_PAYLOAD")
        data = payload.encode() if payload is not None else None
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=float(os.environ["TRP_TIMEOUT"])) as resp:
                print(json.dumps({{"status": resp.status, "body": resp.read().decode()}}))
        except HTTPError as exc:
            print(json.dumps({{"status": exc.code, "body": exc.read().decode()}}))
        except (URLError, OSError) as exc:
            print(json.dumps({{"status": 0, "body": str(exc)}}))
        """
    ).strip()
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        f"container:{container_name}",
        "-e",
        f"TRP_PATH={path}",
        "-e",
        f"TRP_METHOD={method}",
        "-e",
        f"TRP_HEADERS={json.dumps(headers)}",
        "-e",
        f"TRP_TIMEOUT={timeout}",
    ]
    if payload is not None:
        cmd.extend(["-e", f"TRP_PAYLOAD={json.dumps(payload)}"])
    cmd.extend([MOCK_CONTAINER_IMAGE, "python", "-c", script])
    result = run_command(cmd, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown management helper error"
        return 0, detail
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return 0, f"invalid management helper response: {exc}; raw={result.stdout}"
    return int(parsed.get("status", 0)), str(parsed.get("body", ""))


def wait_for_management_api_ready(container_name: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last_status = None
    last_body = ""
    while time.time() < deadline:
        status, body = management_api_request(
            container_name,
            "/api/v1/status",
            method="GET",
            timeout=2.0,
        )
        last_status = status
        last_body = body
        if status == 200:
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"management API did not become ready: status={last_status} body={last_body}"
    )


def bootstrap_tiny_proxy_gateway_scenario(scenario: str, container_name: str) -> str:
    wait_for_management_api_ready(container_name)
    project_id = f"{BENCHMARK_PROJECT_PREFIX}-{scenario}"
    provider_name = (
        "alpha"
        if scenario in PROMPT_CACHE_AFFINITY_SCENARIOS | {"semantic-cache-affinity"}
        else "openai"
    )
    project_status, project_body = management_api_request(
        container_name,
        "/api/v1/projects",
        payload={
            "project_id": project_id,
            "name": f"OSS Gateway Shootout {scenario}",
        },
        method="POST",
    )
    if project_status not in {200, 201}:
        raise RuntimeError(
            f"failed to create benchmark project for {scenario}: status={project_status} body={project_body}"
        )

    if scenario == "tool-round-trip":
        tool_status, tool_body = management_api_request(
            container_name,
            f"/api/v1/projects/{project_id}/tools/tool_echo",
            payload={
                "description": "Echo the benchmark query",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
                "executor_kind": "webhook",
                "executor_config": {
                    "url": mock_gateway_upstream("/tool/echo"),
                    "method": "POST",
                },
                "enabled": True,
                "timeout_ms": 5000,
            },
            method="PUT",
        )
        if tool_status != 200:
            raise RuntimeError(
                f"failed to register benchmark tool: status={tool_status} body={tool_body}"
            )

    key_status, key_body = management_api_request(
        container_name,
        "/api/v1/keys",
        payload={
            "project_id": project_id,
            "name": f"{scenario}-runtime-key",
            "provider_name": provider_name,
        },
        method="POST",
    )
    if key_status != 201:
        raise RuntimeError(
            f"failed to create runtime key for {scenario}: status={key_status} body={key_body}"
        )
    try:
        payload = json.loads(key_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid key-create response: {exc}; body={key_body}") from exc
    key = payload.get("key")
    if not isinstance(key, str) or not key:
        raise RuntimeError(f"runtime key missing from response: {key_body}")
    return key


def upsert_project_policy(container_name: str, project_id: str, payload: dict[str, Any]) -> None:
    status, body = management_api_request(
        container_name,
        f"/api/v1/projects/{project_id}/policy",
        payload=payload,
        method="PUT",
    )
    if status != 200:
        raise RuntimeError(
            f"failed to upsert project policy for {project_id}: status={status} body={body}"
        )


def validate_debug_headers(
    host_port: int,
    variant: RequestVariant,
    *,
    expected_selected_provider: str,
    expected_semantic_cache_status: str | None = None,
    expected_prompt_cache_route: str | None = None,
    expected_semantic_cache_route: str | None = None,
) -> None:
    debug_headers = dict(variant.headers)
    debug_headers["x-trp-routing-debug"] = "1"
    status, headers, body = http_request_detail(
        host_port,
        RequestVariant(path=variant.path, body=variant.body, headers=debug_headers),
        timeout=5.0,
    )
    if status != 200:
        raise RuntimeError(
            f"cache-affinity validation request failed: status={status} body={body}"
        )
    selected = headers.get("x-trp-provider-selected")
    if selected != expected_selected_provider:
        raise RuntimeError(
            f"unexpected selected provider during validation: expected={expected_selected_provider} got={selected!r}"
        )
    if expected_prompt_cache_route is not None:
        route = headers.get("x-trp-prompt-cache-route")
        if route != expected_prompt_cache_route:
            raise RuntimeError(
                f"unexpected prompt-cache route during validation: expected={expected_prompt_cache_route} got={route!r}"
            )
    if expected_semantic_cache_route is not None:
        route = headers.get("x-trp-semantic-cache-route")
        if route != expected_semantic_cache_route:
            raise RuntimeError(
                f"unexpected semantic-cache route during validation: expected={expected_semantic_cache_route} got={route!r}"
            )
    if expected_semantic_cache_status is not None:
        cache_status = headers.get("x-trp-semantic-cache")
        if cache_status != expected_semantic_cache_status:
            raise RuntimeError(
                f"unexpected semantic-cache status during validation: expected={expected_semantic_cache_status} got={cache_status!r}"
            )


def warm_cache_affinity_scenario(
    scenario: str,
    host_port: int,
    runtime_key: str,
    container_name: str,
) -> None:
    project_id = f"{BENCHMARK_PROJECT_PREFIX}-{scenario}"
    if scenario in PROMPT_CACHE_AFFINITY_SCENARIOS:
        upsert_project_policy(
            container_name,
            project_id,
            {
                "fallback_order": ["alpha", "beta"],
                "adaptive_enabled": False,
            },
        )
        warm_variant = RequestVariant(
            path="/v1/chat/completions",
            body={
                "model": "benchmark-model",
                "messages": [{"role": "user", "content": "warm alpha prompt cache"}],
                "trp_prompt_cache": {
                    "enabled": True,
                    "ttl": "24h",
                    "key": "tenant:bench-affinity",
                },
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {runtime_key}",
            },
        )
        status, body = http_request(host_port, warm_variant, timeout=5.0)
        if status != 200:
            raise RuntimeError(
                f"failed to warm prompt-cache affinity scenario: status={status} body={body}"
            )
        upsert_project_policy(
            container_name,
            project_id,
            {
                "fallback_order": ["beta", "alpha"],
                "adaptive_enabled": False,
            },
        )
        benchmark_variant = RequestVariant(
            path="/v1/chat/completions",
            body={
                "model": "benchmark-model",
                "messages": [{"role": "user", "content": "use the warm cache again"}],
                "trp_prompt_cache": {
                    "enabled": True,
                    "ttl": "24h",
                    "key": "tenant:bench-affinity",
                },
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {runtime_key}",
            },
        )
        validate_debug_headers(
            host_port,
            benchmark_variant,
            expected_selected_provider="alpha",
            expected_prompt_cache_route="alpha",
        )
        return

    if scenario == "semantic-cache-affinity":
        upsert_project_policy(
            container_name,
            project_id,
            {
                "fallback_order": ["alpha", "beta"],
                "adaptive_enabled": False,
                "semantic_cache_enabled": True,
                "semantic_cache_ttl_secs": 600,
                "semantic_cache_similarity_threshold": 0.7,
            },
        )
        warm_variant = RequestVariant(
            path="/v1/chat/completions",
            body={
                "model": "benchmark-model",
                "messages": [{"role": "user", "content": "reset password help"}],
                "trp_semantic_cache": {
                    "enabled": True,
                    "ttl_secs": 600,
                    "similarity_threshold": 0.7,
                },
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {runtime_key}",
            },
        )
        status, body = http_request(host_port, warm_variant, timeout=5.0)
        if status != 200:
            raise RuntimeError(
                f"failed to warm semantic-cache affinity scenario: status={status} body={body}"
            )
        upsert_project_policy(
            container_name,
            project_id,
            {
                "fallback_order": ["beta", "alpha"],
                "adaptive_enabled": False,
                "semantic_cache_enabled": True,
                "semantic_cache_ttl_secs": 600,
                "semantic_cache_similarity_threshold": 0.7,
            },
        )
        benchmark_variant = RequestVariant(
            path="/v1/chat/completions",
            body={
                "model": "benchmark-model",
                "messages": [{"role": "user", "content": "need password reset help"}],
                "trp_semantic_cache": {
                    "enabled": True,
                    "ttl_secs": 600,
                    "similarity_threshold": 0.7,
                },
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {runtime_key}",
            },
        )
        validate_debug_headers(
            host_port,
            benchmark_variant,
            expected_selected_provider="alpha",
            expected_semantic_cache_status="hit",
            expected_semantic_cache_route="alpha",
        )


def prepare_tiny_proxy_variant(
    scenario: str,
    variant: RequestVariant,
    container_name: str,
    host_port: int,
) -> RequestVariant:
    runtime_key = bootstrap_tiny_proxy_gateway_scenario(scenario, container_name)
    if scenario in PROMPT_CACHE_AFFINITY_SCENARIOS | {"semantic-cache-affinity"}:
        warm_cache_affinity_scenario(scenario, host_port, runtime_key, container_name)
    headers = dict(variant.headers)
    headers["Authorization"] = f"Bearer {runtime_key}"
    return RequestVariant(path=variant.path, body=variant.body, headers=headers)


def dump_container_logs(container_name: str, destination: pathlib.Path) -> None:
    result = run_command(["docker", "logs", container_name], check=False)
    destination.write_text((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""))


def container_inspect(container_name: str) -> dict[str, Any]:
    result = run_command(["docker", "inspect", container_name])
    parsed = json.loads(result.stdout)
    return parsed[0]


def image_inspect(image_name: str) -> dict[str, Any]:
    result = run_command(["docker", "image", "inspect", image_name])
    parsed = json.loads(result.stdout)
    return parsed[0]


def parse_docker_stats_line(line: str) -> dict[str, float]:
    data = json.loads(line)
    cpu_percent = float(data["CPUPerc"].strip().rstrip("%") or "0")
    rss_bytes = parse_byte_size(data["MemUsage"].split("/")[0].strip())
    return {"cpu_percent": cpu_percent, "rss_bytes": rss_bytes}


class DockerSampler:
    def __init__(self, container_name: str, sample_interval_s: float = 0.5) -> None:
        self.container_name = container_name
        self.sample_interval_s = sample_interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            result = run_command(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{ json . }}",
                    self.container_name,
                ],
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    self.samples.append(parse_docker_stats_line(result.stdout.strip()))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
            if self._stop.wait(self.sample_interval_s):
                return


def parse_hey_summary(output: str) -> dict[str, Any]:
    requests_sec = re.search(r"Requests/sec:\s+([0-9.]+)", output)
    average = re.search(r"Average:\s+([0-9.]+)\s+secs", output)
    total = re.search(r"Total:\s+([0-9.]+)\s+secs", output)
    success = re.search(r"Success rate:\s+([0-9.]+)%", output)
    if not requests_sec or not average:
        raise RuntimeError(f"failed to parse hey output:\n{output}")

    latencies: dict[str, float] = {}
    for percentile in ("50", "95", "99"):
        match = re.search(rf"^\s*{percentile}%\s+in\s+([0-9.]+)\s+secs", output, re.MULTILINE)
        if match:
            latencies[f"p{percentile}"] = float(match.group(1))

    status_dist: dict[str, int] = {}
    for match in re.finditer(r"^\s*\[(\d+)\]\s+(\d+)\s+responses$", output, re.MULTILINE):
        status_dist[match.group(1)] = int(match.group(2))

    return {
        "requests_per_sec": float(requests_sec.group(1)),
        "average_latency_s": float(average.group(1)),
        "total_time_s": float(total.group(1)) if total else None,
        "success_rate_pct": float(success.group(1)) if success else None,
        "latency_percentiles_s": latencies,
        "status_distribution": status_dist,
        "raw_output": truncate_text(output),
    }


def run_hey(duration: str, concurrency: int, host_port: int, variant: RequestVariant, body_file: pathlib.Path) -> dict[str, Any]:
    body_file.write_text(json.dumps(variant.body))
    url = f"http://127.0.0.1:{host_port}{variant.path}"
    cmd = [
        "hey",
        "-z",
        duration,
        "-c",
        str(concurrency),
        "-m",
        "POST",
        "-T",
        variant.headers.get("Content-Type", "application/json"),
        "-D",
        str(body_file),
    ]
    for header_name, header_value in variant.headers.items():
        if header_name.lower() == "content-type":
            continue
        cmd.extend(["-H", f"{header_name}: {header_value}"])
    cmd.append(url)
    result = run_command(cmd)
    return parse_hey_summary(result.stdout)


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, float] | None:
    if not samples:
        return None
    return {
        "avg_cpu_percent": round(statistics.fmean(sample["cpu_percent"] for sample in samples), 3),
        "peak_cpu_percent": round(max(sample["cpu_percent"] for sample in samples), 3),
        "avg_rss_bytes": int(statistics.fmean(sample["rss_bytes"] for sample in samples)),
        "peak_rss_bytes": int(max(sample["rss_bytes"] for sample in samples)),
    }


def render_summary(results: dict[str, Any]) -> str:
    lines = [
        "# OSS Gateway Shootout",
        "",
        f"- Run timestamp: `{results['run_started_at']}`",
        f"- Docker CPU limit per gateway: `{results['limits']['cpus']}`",
        f"- Docker memory limit per gateway: `{results['limits']['memory']}`",
        f"- Scenario: `{results['benchmark']['scenario']}`",
        f"- hey duration: `{results['benchmark']['duration']}`",
        f"- hey concurrency: `{results['benchmark']['concurrency']}`",
        "",
        "| Target | Req/s | Avg (s) | p50 (s) | p95 (s) | p99 (s) | Peak RSS (MiB) | Avg CPU % | Request Path |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in results["target_order"]:
        target = results["targets"].get(name, {})
        summary = target.get("hey_summary") or {}
        samples = target.get("resource_summary") or {}
        peak_rss_mib = samples.get("peak_rss_bytes")
        if peak_rss_mib is not None:
            peak_rss_mib = round(peak_rss_mib / (1024 * 1024), 2)
        lines.append(
            "| {name} | {rps} | {avg} | {p50} | {p95} | {p99} | {rss} | {cpu} | `{path}` |".format(
                name=name,
                rps=summary.get("requests_per_sec", "-"),
                avg=summary.get("average_latency_s", "-"),
                p50=summary.get("latency_percentiles_s", {}).get("p50", "-"),
                p95=summary.get("latency_percentiles_s", {}).get("p95", "-"),
                p99=summary.get("latency_percentiles_s", {}).get("p99", "-"),
                rss=peak_rss_mib if peak_rss_mib is not None else "-",
                cpu=samples.get("avg_cpu_percent", "-"),
                path=target.get("request_variant", {}).get("path", "-"),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `direct` is the mock upstream baseline and has no gateway resource sample.",
            f"- Every benchmark in this run used the `{results['benchmark']['scenario']}` request shape.",
            "- Gateway containers run with the configured Docker log driver, defaulting to `none` to avoid benchmarking stdout collection.",
            "- The mock upstream runs with `--quiet` to avoid per-request logging noise.",
            "- `tiny-proxy` runs with `RUST_LOG=warn`, LiteLLM runs with an explicit quiet `--log_config`, and Bifrost runs with `LOG_LEVEL=error`.",
            "- Config files for every gateway are stored in `configs/` next to this summary.",
            "- Raw `hey` output and Docker inspect metadata are stored in `results.json`.",
        ]
    )
    if results["benchmark"]["scenario"] == "prompt-cache":
        lines.append(
            "- In `prompt-cache`, `tiny-proxy` bootstraps the real gateway prompt-cache runtime and uses a managed runtime key before the benchmark begins."
        )
    if results["benchmark"]["scenario"] == "prompt-cache-affinity":
        lines.append(
            "- In `prompt-cache-affinity`, `tiny-proxy` warms `alpha`, flips fallback order to prefer `beta`, and then benchmarks the durable prompt-cache affinity path. `direct` is the self-routed warmed-provider baseline."
        )
    if results["benchmark"]["scenario"] == "prompt-cache-affinity-routing-only":
        lines.append(
            "- In `prompt-cache-affinity-routing-only`, `tiny-proxy` runs the same warm-and-reroute flow as `prompt-cache-affinity`, but disables durable routing-hint persistence so the result isolates routing overhead from store-write cost."
        )
    if results["benchmark"]["scenario"] == "semantic-cache-affinity":
        lines.append(
            "- In `semantic-cache-affinity`, `tiny-proxy` warms a semantic-cache entry on `alpha`, flips fallback order to prefer `beta`, and then benchmarks the cache-locality routing plus gateway-served cache hit path. `direct` is a Python mock semantic-cache-hit baseline, so this scenario is not a raw proxy-overhead apples-to-apples comparison."
        )
    if results["benchmark"]["scenario"] == "tool-round-trip":
        lines.append(
            "- In `tool-round-trip`, `tiny-proxy` exercises the managed tool runtime; `direct` is a self-orchestrated mock-upstream baseline for the same user-level task."
        )
    for name in results["target_order"]:
        target = results["targets"].get(name, {})
        error = target.get("error")
        if error:
            lines.append(f"- `{name}` failed: {error}")
    return "\n".join(lines) + "\n"


def host_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        mem = run_command(["sysctl", "-n", "hw.memsize"], check=False)
        if mem.returncode == 0 and mem.stdout.strip().isdigit():
            info["memory_bytes"] = int(mem.stdout.strip())
    return info


def benchmark_target(
    target: TargetSpec,
    args: argparse.Namespace,
    config_paths: dict[str, pathlib.Path],
    results_dir: pathlib.Path,
    network_name: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {"name": target.name}
    sampler: DockerSampler | None = None
    container_name: str | None = None

    try:
        if target.name == "direct":
            variant = target.request_variants[0]
            record["request_variant"] = asdict(variant)
            body_file = results_dir / f"{target.name}-request.json"
            record["hey_summary"] = run_hey(
                args.duration, args.concurrency, target.host_port, variant, body_file
            )
            return record

        container_name = start_gateway_container(
            target,
            config_paths,
            network_name,
            args.cpus,
            args.memory,
            args.gateway_log_driver,
            args.scenario,
        )
        inspect_data = container_inspect(container_name)
        image_data = image_inspect(target.image)
        record["container"] = {
            "name": container_name,
            "id": inspect_data["Id"],
            "image": target.image,
            "image_id": image_data["Id"],
            "limits": {
                "cpus": inspect_data["HostConfig"].get("NanoCpus", 0) / 1_000_000_000,
                "memory_bytes": inspect_data["HostConfig"].get("Memory", 0),
            },
        }

        if target.name == "tiny-proxy" and args.scenario in MANAGED_BOOTSTRAP_SCENARIOS:
            variant = prepare_tiny_proxy_variant(
                args.scenario, target.request_variants[0], container_name, target.host_port
            )
            status, body = http_request(target.host_port, variant, timeout=5.0)
            if status != 200:
                raise RuntimeError(
                    f"tiny-proxy bootstrap request failed for {args.scenario}: status={status} body={body}"
                )
        else:
            variant = choose_working_variant(target)
        record["request_variant"] = asdict(variant)

        sampler = DockerSampler(container_name)
        sampler.start()
        body_file = results_dir / f"{target.name}-request.json"
        record["hey_summary"] = run_hey(
            args.duration, args.concurrency, target.host_port, variant, body_file
        )
        sampler.stop()
        record["resource_samples"] = sampler.samples
        record["resource_summary"] = summarize_samples(sampler.samples)
        return record
    except Exception as exc:
        record["error"] = str(exc)
        if container_name:
            log_path = results_dir / f"{target.name}.log"
            dump_container_logs(container_name, log_path)
            record["log_path"] = str(log_path.relative_to(results_dir))
        return record
    finally:
        if sampler is not None:
            sampler.stop()
        if container_name is not None:
            stop_container(container_name)


def render_dry_run(args: argparse.Namespace, config_paths: dict[str, pathlib.Path]) -> None:
    print("Dry run only. Generated configs:")
    for name, path in config_paths.items():
        print(f"\n[{name}] {path}")
        print(path.read_text())

    print("\nPlanned Docker limits:")
    print(f"  cpus={args.cpus}")
    print(f"  memory={args.memory}")
    print(f"  duration={args.duration}")
    print(f"  concurrency={args.concurrency}")
    print(f"  scenario={args.scenario}")
    print(f"  gateway_log_driver={args.gateway_log_driver}")


def main() -> int:
    args = parse_args()
    validate_targets_for_scenario(args.targets, args.scenario)
    results_base = pathlib.Path(args.results_dir)
    results_dir = ensure_results_dir(results_base)
    config_paths = write_configs(results_dir, args.scenario)

    if args.dry_run:
        render_dry_run(args, config_paths)
        return 0

    require_command("docker")
    require_command("hey")
    ensure_docker_available()

    specs = benchmark_targets(args.scenario)
    maybe_pull_images(args, specs)
    if "tiny-proxy" in args.targets:
        ensure_tiny_proxy_image(args.rebuild_tiny_image)

    network_name = f"{NETWORK_NAME}-{os.getpid()}"
    create_network(network_name)
    mock_container = None
    results: dict[str, Any] = {
        "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": host_metadata(),
        "limits": {"cpus": args.cpus, "memory": args.memory},
        "normalization": {
            "gateway_log_driver": args.gateway_log_driver,
            "mock_upstream_quiet": True,
            "tiny_proxy_rust_log": "warn",
            "tiny_proxy_management_bootstrap": args.scenario in MANAGED_BOOTSTRAP_SCENARIOS,
            "litellm_log_config": "warning root with uvicorn.access=CRITICAL",
            "litellm_telemetry": False,
            "bifrost_log_level": "error",
        },
        "benchmark": {
            "scenario": args.scenario,
            "duration": args.duration,
            "duration_seconds": parse_duration_to_seconds(args.duration),
            "concurrency": args.concurrency,
            "supported_targets": sorted(
                {
                    "pass-through": set(DEFAULT_TARGETS),
                    "streaming": set(DEFAULT_TARGETS),
                    "prompt-cache": set(DEFAULT_TARGETS),
                    "prompt-cache-affinity": {"direct", "tiny-proxy"},
                    "prompt-cache-affinity-routing-only": {"direct", "tiny-proxy"},
                    "semantic-cache-affinity": {"direct", "tiny-proxy"},
                    "tool-round-trip": {"direct", "tiny-proxy"},
                }[args.scenario]
            ),
        },
        "target_order": list(args.targets),
        "targets": {},
        "sources": {
            "litellm": "https://docs.litellm.ai/",
            "bifrost": "https://docs.getbifrost.ai/quickstart/gateway/setting-up",
        },
    }

    def _cleanup(*_unused: Any) -> None:
        if mock_container is not None:
            stop_container(mock_container)
        remove_network(network_name)

    previous_sigint = signal.signal(signal.SIGINT, _cleanup)
    previous_sigterm = signal.signal(signal.SIGTERM, _cleanup)

    try:
        mock_container = start_mock_upstream(network_name, specs["direct"].host_port)
        wait_for_host_port(specs["direct"].host_port)

        for target_name in args.targets:
            target = specs[target_name]
            print(f"==> benchmarking {target_name}")
            results["targets"][target_name] = benchmark_target(
                target, args, config_paths, results_dir, network_name
            )

        summary = render_summary(results)
        (results_dir / "SUMMARY.md").write_text(summary)
        (results_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(summary)
        print(f"Artifacts written to {results_dir}")
        return 0
    except Exception as exc:
        results["failure"] = str(exc)
        (results_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        print(f"Partial artifacts written to {results_dir}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        _cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
