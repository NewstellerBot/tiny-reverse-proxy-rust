#!/usr/bin/env python3
"""
Live smoke test for the OpenAI Realtime WebSocket API.

This script can test:
1. a local proxy endpoint, e.g. ws://127.0.0.1:8080/v1/realtime
2. the OpenAI API directly, e.g. wss://api.openai.com/v1/realtime

It supports two scenarios:
- smoke: one request/response round-trip
- workload: several non-trivial text tasks in one conversation

It follows the documented text flow:
- connect and wait for session.created
- send session.update
- send conversation.item.create
- wait for conversation item acknowledgement
- send response.create
- wait for response.done

Examples:
  export OPENAI_API_KEY=...

  # Test a local proxy that forwards /v1/realtime to OpenAI
  uv run --with websockets python3 scripts/openai_realtime_smoke.py \
    --start-proxy --scenario workload

  # Test OpenAI directly
  uv run --with websockets python3 scripts/openai_realtime_smoke.py \
    --url wss://api.openai.com/v1/realtime --scenario smoke
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Callable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class SmokeFailure(RuntimeError):
    pass


@dataclass
class WorkloadTask:
    name: str
    prompt: str
    response_instructions: str
    validator: Callable[[str], None]


@dataclass
class ProxyHandle:
    process: subprocess.Popen[str]
    port: int
    config_path: str
    log_path: str


@dataclass
class TurnResult:
    text: str
    done_event: dict[str, Any]
    elapsed_s: float


def load_websockets():
    try:
        import websockets  # type: ignore
    except ImportError:
        print(
            "missing dependency: run this via "
            "`uv run --with websockets python3 scripts/openai_realtime_smoke.py ...`",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test an OpenAI Realtime WebSocket endpoint."
    )
    parser.add_argument(
        "--scenario",
        choices=("smoke", "workload"),
        default="smoke",
        help="Which scenario to run. Default: smoke",
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8080/v1/realtime",
        help=(
            "Realtime WebSocket URL to test. "
            "Default: ws://127.0.0.1:8080/v1/realtime"
        ),
    )
    parser.add_argument(
        "--start-proxy",
        action="store_true",
        help="Start a local tiny-reverse-proxy process and route /v1/* to OpenAI.",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=18080,
        help="Local proxy port when --start-proxy is used. Default: 18080",
    )
    parser.add_argument(
        "--proxy-binary",
        default="target/debug/tiny-reverse-proxy",
        help="Path to the proxy binary used by --start-proxy.",
    )
    parser.add_argument(
        "--proxy-upstream",
        default="https://api.openai.com",
        help="Upstream route target used by --start-proxy. Default: https://api.openai.com",
    )
    parser.add_argument(
        "--proxy-start-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the local proxy to bind. Default: 10",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        help="Realtime model to request when the URL does not already include ?model=...",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the API key. Default: OPENAI_API_KEY",
    )
    parser.add_argument(
        "--organization-env",
        default="OPENAI_ORGANIZATION",
        help="Optional env var for OpenAI-Organization. Default: OPENAI_ORGANIZATION",
    )
    parser.add_argument(
        "--project-env",
        default="OPENAI_PROJECT",
        help="Optional env var for OpenAI-Project. Default: OPENAI_PROJECT",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("OPENAI_REALTIME_TIMEOUT_SECS", "20")),
        help="Per-event timeout in seconds. Default: 20",
    )
    parser.add_argument(
        "--instructions",
        default=(
            "You are a precise operations assistant. Follow the response instructions "
            "for each turn exactly."
        ),
        help="Base instructions used for session.update.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly PONG.",
        help="User text sent via conversation.item.create.",
    )
    parser.add_argument(
        "--expect-substring",
        default="PONG",
        help="Required substring in the final response text. Default: PONG",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help="Print every sent and received Realtime event.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SmokeFailure(f"required environment variable is missing: {name}")
    return value


def maybe_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    return None


def normalized_url(args: argparse.Namespace) -> str:
    default_url = "ws://127.0.0.1:8080/v1/realtime"
    if args.start_proxy and args.url == default_url:
        return f"ws://127.0.0.1:{args.proxy_port}/v1/realtime"
    return args.url


def with_model_query(url: str, model: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("model", model)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def make_event_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def event_type(event: dict[str, Any]) -> str:
    value = event.get("type")
    if isinstance(value, str):
        return value
    return ""


def format_event(event: dict[str, Any]) -> str:
    return json.dumps(event, separators=(",", ":"), sort_keys=True)


def parse_json_response(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise SmokeFailure(f"expected JSON object in response, got: {text!r}")

    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"failed to parse JSON response: {text!r}") from exc

    if not isinstance(parsed, dict):
        raise SmokeFailure(f"expected JSON object, got: {text!r}")
    return parsed


def extract_output_text(event: dict[str, Any]) -> str:
    parts: list[str] = []
    response = event.get("response")
    if not isinstance(response, dict):
        return ""

    output = response.get("output")
    if not isinstance(output, list):
        return ""

    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_part in content:
            if not isinstance(content_part, dict):
                continue
            text = content_part.get("text")
            if isinstance(text, str):
                parts.append(text)

    return "".join(parts).strip()


async def send_event(ws: Any, event: dict[str, Any], show_events: bool) -> None:
    payload = json.dumps(event)
    if show_events:
        print(f">>> {payload}")
    await ws.send(payload)


async def recv_event(ws: Any, timeout: float, show_events: bool) -> dict[str, Any]:
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout)
        if isinstance(raw, bytes):
            continue

        if show_events:
            print(f"<<< {raw}")

        event = json.loads(raw)
        if not isinstance(event, dict):
            raise SmokeFailure(f"unexpected non-object event: {raw}")
        return event


async def wait_for_event(
    ws: Any, expected_type: str, timeout: float, show_events: bool
) -> dict[str, Any]:
    try:
        while True:
            event = await recv_event(ws, timeout, show_events)
            typ = event_type(event)
            if typ == "error":
                raise SmokeFailure(
                    f"Realtime API returned error event: {format_event(event)}"
                )
            if typ == expected_type:
                return event
    except asyncio.TimeoutError as exc:
        raise SmokeFailure(f"timed out waiting for {expected_type}") from exc


async def wait_for_any_event(
    ws: Any, expected_types: list[str], timeout: float, show_events: bool
) -> dict[str, Any]:
    try:
        while True:
            event = await recv_event(ws, timeout, show_events)
            typ = event_type(event)
            if typ == "error":
                raise SmokeFailure(
                    f"Realtime API returned error event: {format_event(event)}"
                )
            if typ in expected_types:
                return event
    except asyncio.TimeoutError as exc:
        joined = ", ".join(expected_types)
        raise SmokeFailure(f"timed out waiting for one of: {joined}") from exc


async def wait_for_response_text(
    ws: Any, timeout: float, show_events: bool
) -> tuple[str, dict[str, Any]]:
    deltas: list[str] = []

    try:
        while True:
            event = await recv_event(ws, timeout, show_events)
            typ = event_type(event)

            if typ == "error":
                raise SmokeFailure(
                    f"Realtime API returned error event: {format_event(event)}"
                )

            if typ in ("response.output_text.delta", "response.text.delta"):
                delta = event.get("delta")
                if isinstance(delta, str):
                    deltas.append(delta)
                continue

            if typ in ("response.output_text.done", "response.text.done"):
                done_text = event.get("text")
                if isinstance(done_text, str) and not deltas:
                    deltas.append(done_text)
                continue

            if typ == "response.done":
                status = None
                response = event.get("response")
                if isinstance(response, dict):
                    status = response.get("status")

                if status not in (None, "completed"):
                    raise SmokeFailure(
                        "response.done reported non-completed status: "
                        f"{format_event(event)}"
                    )

                text = "".join(deltas).strip()
                if not text:
                    text = extract_output_text(event)
                if not text:
                    raise SmokeFailure(
                        f"response completed without any text output: {format_event(event)}"
                    )
                return text, event
    except asyncio.TimeoutError as exc:
        raise SmokeFailure("timed out waiting for response.done / text output") from exc


def response_usage(done_event: dict[str, Any]) -> dict[str, Any] | None:
    response = done_event.get("response")
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            return usage
    return None


async def run_turn(
    ws: Any,
    prompt: str,
    response_instructions: str,
    timeout: float,
    show_events: bool,
) -> TurnResult:
    started = time.perf_counter()

    print("sending conversation.item.create")
    await send_event(
        ws,
        {
            "event_id": make_event_id("item"),
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        },
        show_events,
    )
    print("waiting for conversation item acknowledgement")
    await wait_for_any_event(
        ws,
        [
            "conversation.item.created",
            "conversation.item.added",
            "conversation.item.done",
        ],
        timeout,
        show_events,
    )

    print("sending response.create")
    await send_event(
        ws,
        {
            "event_id": make_event_id("response"),
            "type": "response.create",
            "response": {
                "instructions": response_instructions,
                "output_modalities": ["text"],
            },
        },
        show_events,
    )

    print("waiting for response.done / text output")
    text, done_event = await wait_for_response_text(ws, timeout, show_events)
    return TurnResult(
        text=text,
        done_event=done_event,
        elapsed_s=time.perf_counter() - started,
    )


def validate_meeting_digest(text: str) -> None:
    lowered = text.lower()
    required = ["sam", "priya", "luis", "redis", "rollback"]
    missing = [value for value in required if value not in lowered]
    if missing:
        raise SmokeFailure(
            f"meeting digest missing expected facts {missing}: {text!r}"
        )


def validate_invoice_json(text: str) -> None:
    parsed = parse_json_response(text)

    if parsed.get("invoice_id") != "INV-24017":
        raise SmokeFailure(f"unexpected invoice_id in {parsed!r}")
    if parsed.get("customer") != "Acme Industrial":
        raise SmokeFailure(f"unexpected customer in {parsed!r}")
    if parsed.get("due_date") != "2026-02-14":
        raise SmokeFailure(f"unexpected due_date in {parsed!r}")

    total = parsed.get("total_usd")
    if total not in (4620, 4620.0, "4620", "4620.00"):
        raise SmokeFailure(f"unexpected total_usd in {parsed!r}")

    items = parsed.get("line_items")
    if not isinstance(items, list) or len(items) < 2:
        raise SmokeFailure(f"expected at least 2 line_items in {parsed!r}")


def validate_invoice_follow_up(text: str) -> None:
    lowered = text.lower()
    required = ["acme", "4620", "overdue"]
    missing = [value for value in required if value not in lowered]
    if missing:
        raise SmokeFailure(
            f"invoice follow-up missing expected facts {missing}: {text!r}"
        )


def workload_tasks() -> list[WorkloadTask]:
    return [
        WorkloadTask(
            name="meeting_digest",
            prompt=(
                "Operations meeting notes:\n"
                "- Sam confirmed the rollback playbook is still manual and needs automation by Mar 12.\n"
                "- Priya owns Redis tuning after cache hit rate fell from 96% to 81%.\n"
                "- Luis will re-run the canary in eu-west-1 after the fix lands.\n"
                "- Risk: if the cache regression persists, checkout p95 may stay above 900ms.\n"
                "Please summarize the situation for an engineering manager."
            ),
            response_instructions=(
                "Return a concise status update in plain text with short bullets. "
                "Mention risks and owners. Do not use a table."
            ),
            validator=validate_meeting_digest,
        ),
        WorkloadTask(
            name="invoice_extraction",
            prompt=(
                "Extract the following invoice into structured data:\n"
                "Invoice ID: INV-24017\n"
                "Customer: Acme Industrial\n"
                "Due date: 2026-02-14\n"
                "Line items:\n"
                "- Analytics seats (12): 1920 USD\n"
                "- Priority support: 2700 USD\n"
                "Total: 4620 USD"
            ),
            response_instructions=(
                "Return compact JSON only with keys invoice_id, customer, due_date, "
                "total_usd, line_items. No markdown fences."
            ),
            validator=validate_invoice_json,
        ),
        WorkloadTask(
            name="invoice_follow_up",
            prompt=(
                "Using the invoice already discussed in this conversation, answer in one "
                "sentence whether it is overdue as of 2026-03-06 and include the customer "
                "name and total amount."
            ),
            response_instructions=(
                "Reply in one sentence. Mention the customer, whether it is overdue, and the total."
            ),
            validator=validate_invoice_follow_up,
        ),
    ]


def write_proxy_config(path: str, port: int, upstream: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f'port = {port}\n\n[paths]\n"/v1/*" = ["{upstream}"]\n'
        )


async def wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last_error = exc
            await asyncio.sleep(0.1)

    raise SmokeFailure(
        f"timed out waiting for proxy on {host}:{port}: {last_error}"
    )


async def start_proxy(args: argparse.Namespace) -> ProxyHandle:
    binary = pathlib.Path(args.proxy_binary)
    if not binary.exists():
        raise SmokeFailure(
            f"proxy binary not found at {binary}; build it first with `cargo build -p tiny-reverse-proxy`"
        )

    fd, config_path = tempfile.mkstemp(prefix="trp-realtime-", suffix=".toml")
    os.close(fd)
    write_proxy_config(config_path, args.proxy_port, args.proxy_upstream)
    log_path = f"{config_path}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    process = subprocess.Popen(
        [str(binary), config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.getcwd(),
    )

    try:
        await wait_for_port("127.0.0.1", args.proxy_port, args.proxy_start_timeout)
    except Exception:
        with contextlib.suppress(Exception):
            process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=5)
        log_file.close()
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            logs = handle.read()
        raise SmokeFailure(
            f"proxy failed to start; config={config_path} log={log_path}\n{logs}"
        )

    log_file.close()
    return ProxyHandle(
        process=process,
        port=args.proxy_port,
        config_path=config_path,
        log_path=log_path,
    )


def stop_proxy(handle: ProxyHandle) -> None:
    if handle.process.poll() is None:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=5)


async def main_async(args: argparse.Namespace) -> int:
    websockets = load_websockets()

    api_key = require_env(args.api_key_env)
    url = with_model_query(normalized_url(args), args.model)

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "X-Client-Request-Id": make_event_id("py-realtime"),
    }

    organization = maybe_env(args.organization_env)
    if organization:
        headers["OpenAI-Organization"] = organization

    project = maybe_env(args.project_env)
    if project:
        headers["OpenAI-Project"] = project

    connect_kwargs: dict[str, Any] = {
        "open_timeout": args.timeout,
        "close_timeout": args.timeout,
        "max_size": 16 * 1024 * 1024,
    }

    signature = inspect.signature(websockets.connect)
    if "additional_headers" in signature.parameters:
        connect_kwargs["additional_headers"] = headers
    else:
        connect_kwargs["extra_headers"] = headers

    proxy_handle: ProxyHandle | None = None
    if args.start_proxy:
        print(
            f"starting local proxy on 127.0.0.1:{args.proxy_port} -> {args.proxy_upstream}"
        )
        proxy_handle = await start_proxy(args)
        print(
            f"proxy_started config={proxy_handle.config_path} log={proxy_handle.log_path}"
        )

    try:
        print(f"connecting to {url}")

        async with websockets.connect(url, **connect_kwargs) as ws:
            print("waiting for session.created")
            session_created = await wait_for_event(
                ws, "session.created", args.timeout, args.show_events
            )
            session = session_created.get("session")
            if isinstance(session, dict):
                session_id = session.get("id")
                if session_id:
                    print(f"session_created id={session_id}")

            print("sending session.update")
            await send_event(
                ws,
                {
                    "event_id": make_event_id("session"),
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": args.instructions,
                        "output_modalities": ["text"],
                    },
                },
                args.show_events,
            )
            print("waiting for session.updated")
            await wait_for_event(ws, "session.updated", args.timeout, args.show_events)

            if args.scenario == "smoke":
                result = await run_turn(
                    ws,
                    prompt=args.prompt,
                    response_instructions=args.instructions,
                    timeout=args.timeout,
                    show_events=args.show_events,
                )
                print(f"response_text={result.text!r}")

                if args.expect_substring and args.expect_substring not in result.text:
                    raise SmokeFailure(
                        f"expected substring {args.expect_substring!r} in response, "
                        f"got {result.text!r}; response.done={format_event(result.done_event)}"
                    )

                print(
                    f"PASS scenario=smoke elapsed_s={result.elapsed_s:.2f}"
                )
                return 0

            total_tokens = 0
            tasks = workload_tasks()
            for index, task in enumerate(tasks, start=1):
                print(f"task {index}/{len(tasks)}: {task.name}")
                result = await run_turn(
                    ws,
                    prompt=task.prompt,
                    response_instructions=task.response_instructions,
                    timeout=args.timeout,
                    show_events=args.show_events,
                )
                task.validator(result.text)
                usage = response_usage(result.done_event)
                if usage and isinstance(usage.get("total_tokens"), int):
                    total_tokens += usage["total_tokens"]

                print(
                    f"task_pass name={task.name} elapsed_s={result.elapsed_s:.2f} "
                    f"text={result.text!r}"
                )

            print(
                f"PASS scenario=workload tasks={len(tasks)} total_tokens={total_tokens}"
            )
            return 0
    finally:
        if proxy_handle is not None:
            stop_proxy(proxy_handle)


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except SmokeFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
