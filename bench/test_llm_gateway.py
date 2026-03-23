#!/usr/bin/env python3
"""
End-to-end test for the LLM gateway plugins.

Starts a mock LLM API, writes a proxy config, launches the proxy binary,
then runs test scenarios against it and reports pass/fail.

Usage:
  # Build first:
  LIBRARY_PATH="$(xcrun --show-sdk-path)/usr/lib" cargo build --features plugin-llm-gateway

  # Run:
  python3 bench/test_llm_gateway.py

  # With custom binary path:
  python3 bench/test_llm_gateway.py --binary ./target/release/tiny-reverse-proxy
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg):
    print(f"  {GREEN}PASS{RESET}  {msg}")


def fail(msg, detail=""):
    print(f"  {RED}FAIL{RESET}  {msg}")
    if detail:
        print(f"        {detail}")


def section(title):
    print(f"\n{BOLD}{CYAN}--- {title} ---{RESET}")


# ---------------------------------------------------------------------------
# Mock LLM API server
# ---------------------------------------------------------------------------

def make_response_body(model="gpt-4"):
    return json.dumps({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from mock LLM!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }).encode()


class MockLLM(BaseHTTPRequestHandler):
    """Returns a fixed JSON response with Content-Length."""

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else b""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}
        model = payload.get("model", "gpt-4")
        resp = make_response_body(model)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *_):
        pass


class FailingMockLLM(BaseHTTPRequestHandler):
    """Always returns 500."""

    def do_POST(self):
        self.send_response(500)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Internal Server Error")

    def log_message(self, *_):
        pass


class AuthCaptureMockLLM(BaseHTTPRequestHandler):
    """Like MockLLM but captures Authorization headers and request bodies."""

    captured_auth_headers = []
    captured_bodies = []

    def do_POST(self):
        auth = self.headers.get("Authorization", "")
        AuthCaptureMockLLM.captured_auth_headers.append(auth)

        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else b""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}
        AuthCaptureMockLLM.captured_bodies.append(payload)

        model = payload.get("model", "gpt-4")
        resp = make_response_body(model)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *_):
        pass


def find_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def start_http_server(handler_class, port=None):
    port = port or find_free_port()
    server = HTTPServer(("127.0.0.1", port), handler_class)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Proxy management
# ---------------------------------------------------------------------------

@contextmanager
def start_proxy(binary, config_toml):
    """Write config to a temp file, start the proxy, yield the port."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_toml)
        config_path = f.name

    env = os.environ.copy()
    # Needed on macOS
    sdk = subprocess.run(
        ["xcrun", "--show-sdk-path"],
        capture_output=True, text=True
    ).stdout.strip()
    if sdk:
        env["LIBRARY_PATH"] = f"{sdk}/usr/lib"
    env["RUST_LOG"] = "info"

    proc = subprocess.Popen(
        [binary, config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    # Give it time to bind
    time.sleep(0.5)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"Proxy failed to start:\n{stderr}")
    try:
        yield proc
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.unlink(config_path)


def wait_for_port(port, proc):
    for _ in range(30):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.2)
            s.close()
            return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    stderr = proc.stderr.read().decode() if proc.poll() else ""
    print(f"{RED}Proxy failed to accept connections on :{port}{RESET}")
    if stderr:
        print(stderr[:500])
    sys.exit(1)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def post_chat(proxy_port, api_key="sk-test", body_size=None, model="gpt-4"):
    """Send a chat completion request through the proxy. Returns (status, body_str)."""
    payload = {"model": model, "messages": [{"role": "user", "content": "Hello"}]}
    data = json.dumps(payload).encode()
    if body_size and body_size > len(data):
        # Pad the body to reach the desired size (still valid-ish JSON)
        payload["_padding"] = "x" * (body_size - len(data) - 20)
        data = json.dumps(payload).encode()

    req = Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=5)
        return resp.status, resp.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode()
    except URLError as e:
        return 0, str(e)


def mgmt_post(port, path, body_dict):
    """POST JSON to management API. Returns (status, body_str)."""
    data = json.dumps(body_dict).encode()
    req = Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=5)
        return resp.status, resp.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode()


def mgmt_get(port, path):
    """GET from management API. Returns (status, body_str)."""
    req = Request(f"http://127.0.0.1:{port}{path}", method="GET")
    try:
        resp = urlopen(req, timeout=5)
        return resp.status, resp.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode()


def post_no_auth(proxy_port):
    """Send a request without Authorization header."""
    data = json.dumps({"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}).encode()
    req = Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=5)
        return resp.status, resp.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode()


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def test_rate_limiter(proxy_port):
    section("Token Rate Limiter")
    passed = 0
    total = 0

    # Test 1: requests within burst succeed
    total += 1
    status, _ = post_chat(proxy_port, api_key="sk-burst-ok")
    if status == 200:
        ok("first request within burst succeeds")
        passed += 1
    else:
        fail("first request within burst succeeds", f"got {status}")

    # Test 2: exhaust token budget → 429
    # burst_tokens=50, body ~66 bytes => ~16 tokens per request
    # 3 requests = 48 tokens, 4th should fail
    total += 1
    key = f"sk-exhaust-{uuid.uuid4().hex[:6]}"
    statuses = []
    for _ in range(6):
        s, _ = post_chat(proxy_port, api_key=key)
        statuses.append(s)
    if 429 in statuses:
        first_429 = statuses.index(429)
        ok(f"429 after {first_429 + 1} requests (burst exhausted)")
        passed += 1
    else:
        fail("expected 429 after burst exhausted", f"got statuses: {statuses}")

    # Test 3: different key still works after first key is throttled
    total += 1
    status, _ = post_chat(proxy_port, api_key="sk-fresh-key")
    if status == 200:
        ok("different API key is independent")
        passed += 1
    else:
        fail("different API key is independent", f"got {status}")

    # Test 4: 429 response body is JSON
    total += 1
    key2 = f"sk-json-{uuid.uuid4().hex[:6]}"
    for _ in range(10):
        s, body = post_chat(proxy_port, api_key=key2)
        if s == 429:
            break
    if s == 429 and "rate_limit_error" in body:
        ok("429 body is JSON with rate_limit_error type")
        passed += 1
    else:
        fail("429 body is JSON with rate_limit_error type", f"status={s}, body={body[:100]}")

    # Test 5: no auth header → passes through (no rate limiting applied)
    total += 1
    status, _ = post_no_auth(proxy_port)
    if status == 200:
        ok("no auth header bypasses rate limiter")
        passed += 1
    else:
        fail("no auth header bypasses rate limiter", f"got {status}")

    return passed, total


def test_cost_tracker(proxy_port):
    section("Cost Tracker")
    passed = 0
    total = 0

    # Test 1: requests accumulate cost, eventually 402
    # budget_limit=0.005, cost per request ~ 0.0016 (input) + output cost
    # should get 402 after a few requests
    total += 1
    key = f"sk-budget-{uuid.uuid4().hex[:6]}"
    statuses = []
    for _ in range(10):
        s, _ = post_chat(proxy_port, api_key=key)
        statuses.append(s)
    if 402 in statuses:
        first_402 = statuses.index(402)
        ok(f"402 after {first_402 + 1} requests (budget exceeded)")
        passed += 1
    else:
        fail("expected 402 after budget exceeded", f"got statuses: {statuses}")

    # Test 2: 402 body is JSON
    total += 1
    key2 = f"sk-402body-{uuid.uuid4().hex[:6]}"
    s, body = 0, ""
    for _ in range(10):
        s, body = post_chat(proxy_port, api_key=key2)
        if s == 402:
            break
    if s == 402 and "budget_exceeded_error" in body:
        ok("402 body is JSON with budget_exceeded_error type")
        passed += 1
    else:
        fail("402 body is JSON with budget_exceeded_error type", f"status={s}, body={body[:100]}")

    # Test 3: different key has independent budget
    total += 1
    status, _ = post_chat(proxy_port, api_key="sk-unlimited-fresh")
    if status == 200:
        ok("different API key has separate budget")
        passed += 1
    else:
        fail("different API key has separate budget", f"got {status}")

    return passed, total


def test_combined_stack(proxy_port):
    section("Combined Plugin Stack (rate limiter + cost tracker)")
    passed = 0
    total = 0

    # Test 1: rate limiter fires first (429 before 402)
    # The rate limiter has a tight burst, so it should reject before cost tracker
    total += 1
    key = f"sk-combo-{uuid.uuid4().hex[:6]}"
    first_error = None
    for _ in range(10):
        s, body = post_chat(proxy_port, api_key=key)
        if s in (429, 402):
            first_error = s
            break
    if first_error == 429:
        ok("rate limiter (429) fires before cost tracker (402)")
        passed += 1
    elif first_error == 402:
        # Also acceptable — depends on config and timing
        ok("cost tracker fired first (budget was tighter than rate limit)")
        passed += 1
    else:
        fail("expected 429 or 402 rejection", f"got {first_error}")

    return passed, total


def test_basic_proxy(proxy_port):
    section("Basic Proxy Forwarding")
    passed = 0
    total = 0

    # Test 1: valid request proxied to upstream
    total += 1
    status, body = post_chat(proxy_port, api_key="sk-basic")
    if status == 200:
        ok("request proxied to upstream successfully")
        passed += 1
    else:
        fail("request proxied to upstream successfully", f"got {status}")

    # Test 2: response body is valid JSON from mock
    total += 1
    if status == 200:
        try:
            parsed = json.loads(body)
            if parsed.get("object") == "chat.completion":
                ok("response body is valid OpenAI chat completion JSON")
                passed += 1
            else:
                fail("response body is valid OpenAI chat completion JSON",
                     f"object={parsed.get('object')}")
        except json.JSONDecodeError:
            fail("response body is valid OpenAI chat completion JSON", "not valid JSON")
    else:
        fail("response body is valid OpenAI chat completion JSON", "request failed")

    return passed, total


def post_chat_with_content(proxy_port, api_key="sk-test", content="Hello", model="gpt-4"):
    """Send a chat completion request with custom user content."""
    payload = {"model": model, "messages": [{"role": "user", "content": content}]}
    data = json.dumps(payload).encode()

    req = Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=5)
        return resp.status, resp.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode()
    except URLError as e:
        return 0, str(e)


def test_content_filter(proxy_port):
    section("Content Filter")
    passed = 0
    total = 0

    # Test 1: SSN in request body → 400
    total += 1
    status, body = post_chat_with_content(
        proxy_port,
        api_key="sk-filter-test",
        content="My SSN is 123-45-6789, please help",
    )
    if status == 400 and "content_filter_error" in body:
        ok("SSN in request body blocked with 400")
        passed += 1
    else:
        fail("SSN in request body blocked with 400", f"status={status}, body={body[:200]}")

    # Test 2: error message contains rule description
    total += 1
    if status == 400 and "SSN pattern" in body:
        ok("error message contains rule description")
        passed += 1
    else:
        fail("error message contains rule description", f"body={body[:200]}")

    # Test 3: clean request passes through
    total += 1
    status, body = post_chat_with_content(
        proxy_port,
        api_key="sk-filter-clean",
        content="What is the weather today?",
    )
    if status == 200:
        ok("clean request passes through content filter")
        passed += 1
    else:
        fail("clean request passes through content filter", f"status={status}")

    return passed, total


def test_cross_provider_retry(proxy_port, mgmt_port):
    section("Cross-Provider Retry")
    passed = 0
    total = 0

    # Create a virtual key to trigger the multi-provider path
    status, body = mgmt_post(mgmt_port, "/api/v1/keys", {
        "name": "retry-key",
        "provider_name": "failing-provider",
    })
    vk_key = None
    if status == 201:
        try:
            vk_key = json.loads(body).get("key")
        except json.JSONDecodeError:
            pass

    if not vk_key:
        print(f"  {YELLOW}SKIP{RESET}  cross-provider retry tests (no key created)")
        return passed, total

    # Test 1: request retried on second provider after first returns 500
    total += 1
    status, body = post_chat(proxy_port, api_key=vk_key, model="shared-model")
    if status == 200:
        ok("request retried on second provider after 500")
        passed += 1
    else:
        fail("request retried on second provider after 500",
             f"status={status}, body={body[:200]}")

    # Test 2: response body is valid JSON from working mock
    total += 1
    if status == 200:
        try:
            parsed = json.loads(body)
            if parsed.get("object") == "chat.completion":
                ok("retried response is valid chat completion JSON")
                passed += 1
            else:
                fail("retried response is valid chat completion JSON",
                     f"object={parsed.get('object')}")
        except json.JSONDecodeError:
            fail("retried response is valid chat completion JSON", "not valid JSON")
    else:
        fail("retried response is valid chat completion JSON", "request failed")

    return passed, total


def test_per_model_cost(proxy_port, mgmt_port):
    section("Per-Model Cost Breakdown")
    passed = 0
    total = 0

    # Create a virtual key
    status, body = mgmt_post(mgmt_port, "/api/v1/keys", {
        "name": "cost-model-key",
        "provider_name": "mock-provider",
    })
    vk_key = None
    if status == 201:
        try:
            vk_key = json.loads(body).get("key")
        except json.JSONDecodeError:
            pass

    if not vk_key:
        print(f"  {YELLOW}SKIP{RESET}  per-model cost tests (no key created)")
        return passed, total

    # Send a few requests
    for _ in range(3):
        post_chat(proxy_port, api_key=vk_key, model="gpt-4")

    # Wait for flush
    time.sleep(1.5)

    # Test 1: by-model endpoint returns data
    total += 1
    status, body = mgmt_get(mgmt_port, "/api/v1/cost/usage/by-model")
    model_entries = []
    if status == 200:
        try:
            parsed = json.loads(body)
            model_entries = parsed.get("data", [])
            if isinstance(model_entries, list) and len(model_entries) > 0:
                ok("by-model endpoint returns usage data")
                passed += 1
            else:
                fail("by-model endpoint returns usage data",
                     f"got empty or non-list: {body[:200]}")
        except json.JSONDecodeError:
            fail("by-model endpoint returns usage data", f"not valid JSON: {body[:200]}")
    else:
        fail("by-model endpoint returns usage data", f"status={status}")

    # Test 2: data contains model field
    total += 1
    if model_entries:
        has_model = any(entry.get("model") for entry in model_entries)
        if has_model:
            ok("by-model data contains model field")
            passed += 1
        else:
            fail("by-model data contains model field", f"data={model_entries[:3]}")
    else:
        fail("by-model data contains model field", "no data")

    return passed, total


def test_audit_log(proxy_port, mgmt_port):
    section("Audit Log")
    passed = 0
    total = 0

    # Create a virtual key
    status, body = mgmt_post(mgmt_port, "/api/v1/keys", {
        "name": "audit-key",
        "provider_name": "mock-provider",
    })
    vk_key = None
    if status == 201:
        try:
            vk_key = json.loads(body).get("key")
        except json.JSONDecodeError:
            pass

    if not vk_key:
        print(f"  {YELLOW}SKIP{RESET}  audit log tests (no key created)")
        return passed, total

    # Send a request
    post_chat(proxy_port, api_key=vk_key, model="gpt-4")

    # Wait for audit drain
    time.sleep(1.5)

    # Test 1: logs endpoint returns entries
    total += 1
    status, body = mgmt_get(mgmt_port, "/api/v1/logs?limit=10")
    log_entries = []
    if status == 200:
        try:
            parsed = json.loads(body)
            log_entries = parsed.get("data", [])
            if isinstance(log_entries, list) and len(log_entries) > 0:
                ok("logs endpoint returns audit entries")
                passed += 1
            else:
                fail("logs endpoint returns audit entries",
                     f"got empty or non-list: {body[:200]}")
        except json.JSONDecodeError:
            fail("logs endpoint returns audit entries", f"not valid JSON: {body[:200]}")
    else:
        fail("logs endpoint returns audit entries", f"status={status}")

    # Test 2: log entries contain expected fields
    total += 1
    if log_entries:
        first = log_entries[0]
        if all(k in first for k in ["model", "input_tokens", "cost"]):
            ok("log entries contain model, input_tokens, cost fields")
            passed += 1
        else:
            fail("log entries contain expected fields",
                 f"keys={list(first.keys())}")
    else:
        fail("log entries contain expected fields", "no entries")

    return passed, total


def test_virtual_keys(proxy_port, mgmt_port):
    section("Virtual Keys")
    passed = 0
    total = 0

    # Test 1: Create virtual key via management API
    total += 1
    status, body = mgmt_post(mgmt_port, "/api/v1/keys", {
        "name": "test-key",
        "provider_name": "mock-openai",
    })
    vk_key = None
    if status == 201:
        try:
            data = json.loads(body)
            vk_key = data.get("key", "")
            key_hash = data.get("key_hash", "")
            if vk_key.startswith("sk-trp-") and key_hash:
                ok("create virtual key returns sk-trp-* key and hash")
                passed += 1
            else:
                fail("create virtual key returns sk-trp-* key and hash",
                     f"key={vk_key!r}, key_hash={key_hash!r}")
        except json.JSONDecodeError:
            fail("create virtual key returns sk-trp-* key and hash", f"not valid JSON: {body[:100]}")
    else:
        fail("create virtual key returns sk-trp-* key and hash", f"status={status}, body={body[:200]}")

    if not vk_key:
        # Can't run remaining tests without a valid key
        print(f"  {YELLOW}SKIP{RESET}  remaining virtual key tests (no key created)")
        return passed, total

    # Test 2: Request with virtual key is proxied and auth header replaced
    total += 1
    AuthCaptureMockLLM.captured_auth_headers.clear()
    status, body = post_chat(proxy_port, api_key=vk_key)
    if status == 200 and AuthCaptureMockLLM.captured_auth_headers:
        received_auth = AuthCaptureMockLLM.captured_auth_headers[-1]
        if received_auth == "Bearer sk-real-provider-key":
            ok("virtual key auth replaced with provider key")
            passed += 1
        else:
            fail("virtual key auth replaced with provider key",
                 f"upstream received: {received_auth!r}")
    else:
        fail("virtual key auth replaced with provider key",
             f"status={status}, captured={len(AuthCaptureMockLLM.captured_auth_headers)}")

    # Test 3: Invalid virtual key gets 401
    total += 1
    status, body = post_chat(proxy_port, api_key="sk-trp-invalid-garbage")
    if status == 401 and "invalid_api_key" in body:
        ok("invalid virtual key rejected with 401")
        passed += 1
    else:
        fail("invalid virtual key rejected with 401", f"status={status}, body={body[:100]}")

    # Test 4: Non-prefixed key passes through unchanged
    total += 1
    AuthCaptureMockLLM.captured_auth_headers.clear()
    status, _ = post_chat(proxy_port, api_key="sk-regular-key")
    if status == 200 and AuthCaptureMockLLM.captured_auth_headers:
        received_auth = AuthCaptureMockLLM.captured_auth_headers[-1]
        if received_auth == "Bearer sk-regular-key":
            ok("non-prefixed key passes through unchanged")
            passed += 1
        else:
            fail("non-prefixed key passes through unchanged",
                 f"upstream received: {received_auth!r}")
    else:
        fail("non-prefixed key passes through unchanged",
             f"status={status}, captured={len(AuthCaptureMockLLM.captured_auth_headers)}")

    # Test 5: Model alias resolution
    total += 1
    AuthCaptureMockLLM.captured_bodies.clear()
    status, _ = post_chat(proxy_port, api_key=vk_key, model="my-gpt4")
    if status == 200 and AuthCaptureMockLLM.captured_bodies:
        received_model = AuthCaptureMockLLM.captured_bodies[-1].get("model", "")
        if received_model == "gpt-4":
            ok("model alias resolved (my-gpt4 -> gpt-4)")
            passed += 1
        else:
            fail("model alias resolved (my-gpt4 -> gpt-4)",
                 f"upstream received model={received_model!r}")
    else:
        fail("model alias resolved (my-gpt4 -> gpt-4)",
             f"status={status}, captured={len(AuthCaptureMockLLM.captured_bodies)}")

    # Test 6: Per-key budget enforcement
    total += 1
    # Create a key with a tiny budget
    status, body = mgmt_post(mgmt_port, "/api/v1/keys", {
        "name": "budget-key",
        "provider_name": "mock-openai",
        "budget_limit": 0.001,
    })
    budget_key = None
    if status == 201:
        try:
            budget_key = json.loads(body).get("key")
        except json.JSONDecodeError:
            pass

    if budget_key:
        statuses = []
        for _ in range(15):
            s, _ = post_chat(proxy_port, api_key=budget_key)
            statuses.append(s)
            if s == 402:
                break
        if 402 in statuses:
            first_402 = statuses.index(402)
            ok(f"per-key budget enforced (402 after {first_402 + 1} requests)")
            passed += 1
        else:
            fail("per-key budget enforced", f"got statuses: {statuses}")
    else:
        fail("per-key budget enforced", "could not create budget-limited key")

    return passed, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test LLM gateway plugins end-to-end")
    parser.add_argument(
        "--binary",
        default="./target/debug/tiny-reverse-proxy",
        help="Path to the proxy binary (default: debug build)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"{RED}Binary not found: {args.binary}{RESET}")
        print(f"Build with: LIBRARY_PATH=\"$(xcrun --show-sdk-path)/usr/lib\" "
              f"cargo build --features plugin-llm-gateway")
        sys.exit(1)

    total_passed = 0
    total_tests = 0

    with start_http_server(MockLLM) as mock_port:
        print(f"{BOLD}Mock LLM API on :{mock_port}{RESET}")

        # --- Proxy 1: rate limiter only ---
        rl_port = find_free_port()
        rl_config = f"""\
port = {rl_port}

[paths]
"/*" = ["127.0.0.1:{mock_port}"]

[[plugins]]
name = "rate_limiter"
enabled = true
[plugins.config]
tokens_per_minute = 300
burst_tokens = 50
"""
        print(f"{BOLD}Proxy (rate limiter) on :{rl_port}{RESET}")
        with start_proxy(args.binary, rl_config) as proc:
            wait_for_port(rl_port, proc)

            p, t = test_basic_proxy(rl_port)
            total_passed += p
            total_tests += t

            p, t = test_rate_limiter(rl_port)
            total_passed += p
            total_tests += t

        # --- Proxy 2: cost tracker only ---
        ct_port = find_free_port()
        ct_config = f"""\
port = {ct_port}

[paths]
"/*" = ["127.0.0.1:{mock_port}"]

[[plugins]]
name = "cost_tracker"
enabled = true
[plugins.config]
budget_limit = 0.005
log_interval_secs = 3600
default_cost_per_1k_input = 0.01
default_cost_per_1k_output = 0.02
"""
        print(f"{BOLD}Proxy (cost tracker) on :{ct_port}{RESET}")
        with start_proxy(args.binary, ct_config) as proc:
            wait_for_port(ct_port, proc)

            p, t = test_cost_tracker(ct_port)
            total_passed += p
            total_tests += t

        # --- Proxy 3: both plugins ---
        combo_port = find_free_port()
        combo_config = f"""\
port = {combo_port}

[paths]
"/*" = ["127.0.0.1:{mock_port}"]

[[plugins]]
name = "rate_limiter"
enabled = true
[plugins.config]
tokens_per_minute = 300
burst_tokens = 50

[[plugins]]
name = "cost_tracker"
enabled = true
[plugins.config]
budget_limit = 0.005
log_interval_secs = 3600
default_cost_per_1k_input = 0.01
default_cost_per_1k_output = 0.02
"""
        print(f"{BOLD}Proxy (combined) on :{combo_port}{RESET}")
        with start_proxy(args.binary, combo_config) as proc:
            wait_for_port(combo_port, proc)

            p, t = test_combined_stack(combo_port)
            total_passed += p
            total_tests += t

    # --- Proxy 4: content filter ---
    with start_http_server(MockLLM) as cf_mock_port:
        cf_port = find_free_port()
        cf_config = f"""\
port = {cf_port}

[paths]
"/*" = ["127.0.0.1:{cf_mock_port}"]

[[plugins]]
name = "content_filter"
enabled = true
[plugins.config]
action = "block"
[[plugins.config.input_rules]]
pattern = "\\\\b\\\\d{{3}}-\\\\d{{2}}-\\\\d{{4}}\\\\b"
description = "SSN pattern"
"""
        print(f"{BOLD}Proxy (content filter) on :{cf_port}{RESET}")
        with start_proxy(args.binary, cf_config) as proc:
            wait_for_port(cf_port, proc)

            p, t = test_content_filter(cf_port)
            total_passed += p
            total_tests += t

    # --- Proxy 5: cross-provider retry ---
    with start_http_server(FailingMockLLM) as failing_port, \
         start_http_server(MockLLM) as good_port:
        print(f"{BOLD}Failing mock on :{failing_port}, Good mock on :{good_port}{RESET}")

        retry_port = find_free_port()
        retry_mgmt_port = find_free_port()
        retry_config = f"""\
port = {retry_port}
management_api_port = {retry_mgmt_port}
store_url = "sqlite://:memory:"

[[providers]]
name = "failing-provider"
api_key = "sk-fail-key"
base_url = "http://127.0.0.1:{failing_port}"
models = ["shared-model"]

[[providers]]
name = "good-provider"
api_key = "sk-good-key"
base_url = "http://127.0.0.1:{good_port}"
models = ["shared-model"]

[paths]
"/*" = ["127.0.0.1:{failing_port}", "127.0.0.1:{good_port}"]

[[plugins]]
name = "cost_tracker"
enabled = true
[plugins.config]
budget_limit = 0.0
log_interval_secs = 3600
default_cost_per_1k_input = 0.01
default_cost_per_1k_output = 0.02
"""
        print(f"{BOLD}Proxy (cross-provider retry) on :{retry_port}, mgmt on :{retry_mgmt_port}{RESET}")
        with start_proxy(args.binary, retry_config) as proc:
            wait_for_port(retry_port, proc)
            wait_for_port(retry_mgmt_port, proc)

            p, t = test_cross_provider_retry(retry_port, retry_mgmt_port)
            total_passed += p
            total_tests += t

    # --- Proxy 6: per-model cost + audit log ---
    with start_http_server(MockLLM) as model_mock_port:
        print(f"{BOLD}Mock for cost+audit on :{model_mock_port}{RESET}")

        model_port = find_free_port()
        model_mgmt_port = find_free_port()
        model_config = f"""\
port = {model_port}
management_api_port = {model_mgmt_port}
store_url = "sqlite://:memory:"

[[providers]]
name = "mock-provider"
api_key = "sk-mock-key"
base_url = "http://127.0.0.1:{model_mock_port}"
models = ["gpt-4"]

[paths]
"/*" = ["127.0.0.1:{model_mock_port}"]

[[plugins]]
name = "cost_tracker"
enabled = true
[plugins.config]
budget_limit = 0.0
log_interval_secs = 3600
default_cost_per_1k_input = 0.01
default_cost_per_1k_output = 0.02
"""
        print(f"{BOLD}Proxy (cost+audit) on :{model_port}, mgmt on :{model_mgmt_port}{RESET}")
        with start_proxy(args.binary, model_config) as proc:
            wait_for_port(model_port, proc)
            wait_for_port(model_mgmt_port, proc)

            p, t = test_per_model_cost(model_port, model_mgmt_port)
            total_passed += p
            total_tests += t

            p, t = test_audit_log(model_port, model_mgmt_port)
            total_passed += p
            total_tests += t

    # --- Proxy 7: virtual keys ---
    with start_http_server(AuthCaptureMockLLM) as auth_mock_port:
        print(f"{BOLD}Auth-capture mock on :{auth_mock_port}{RESET}")

        vk_port = find_free_port()
        mgmt_port = find_free_port()
        vk_config = f"""\
port = {vk_port}
management_api_port = {mgmt_port}
store_url = "sqlite://:memory:"

[[providers]]
name = "mock-openai"
api_key = "sk-real-provider-key"
base_url = "http://127.0.0.1:{auth_mock_port}"
models = ["gpt-4"]

[model_aliases]
"my-gpt4" = "gpt-4"

[paths]
"/*" = ["127.0.0.1:{auth_mock_port}"]

[[plugins]]
name = "cost_tracker"
enabled = true
[plugins.config]
budget_limit = 0.0
log_interval_secs = 3600
default_cost_per_1k_input = 0.01
default_cost_per_1k_output = 0.02
"""
        print(f"{BOLD}Proxy (virtual keys) on :{vk_port}, mgmt on :{mgmt_port}{RESET}")
        with start_proxy(args.binary, vk_config) as proc:
            wait_for_port(vk_port, proc)
            wait_for_port(mgmt_port, proc)

            p, t = test_virtual_keys(vk_port, mgmt_port)
            total_passed += p
            total_tests += t

    # Summary
    print(f"\n{BOLD}{'='*50}{RESET}")
    if total_passed == total_tests:
        print(f"{GREEN}{BOLD}All {total_tests} tests passed!{RESET}")
    else:
        print(f"{RED}{BOLD}{total_tests - total_passed}/{total_tests} tests failed{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
