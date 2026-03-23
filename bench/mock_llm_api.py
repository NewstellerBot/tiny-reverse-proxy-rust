#!/usr/bin/env python3
"""
Mock LLM API server for manual testing of the LLM gateway plugins.

Supports:
  - POST /v1/chat/completions  (OpenAI-style)
  - POST /v1/messages          (Anthropic-style)
  - SSE streaming via ?stream=true or {"stream":true} in body
  - Configurable latency via --latency flag

Usage:
  python3 bench/mock_llm_api.py                    # default port 9900
  python3 bench/mock_llm_api.py --port 9901        # custom port
  python3 bench/mock_llm_api.py --latency 0.5      # 500ms per chunk

Then configure the proxy to forward to http://127.0.0.1:9900 and test
with the plugin-llm-gateway feature enabled.

Example curl through the proxy:
  curl -X POST http://localhost:8080/v1/chat/completions \\
    -H "Authorization: Bearer sk-test-key" \\
    -H "Content-Type: application/json" \\
    -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def make_openai_response(
    model="gpt-4",
    prompt_cache_enabled=False,
    content="Hello! I'm a mock LLM response for testing the gateway plugins.",
):
    usage = {
        "prompt_tokens": 12,
        "completion_tokens": 15,
        "total_tokens": 27,
    }
    if prompt_cache_enabled:
        usage["prompt_tokens_details"] = {
            "cached_tokens": 64,
            "cache_write_tokens": 16,
        }
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def make_anthropic_response(model="claude-3-opus-20240229"):
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": "Hello! I'm a mock LLM response for testing the gateway plugins.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 15,
        },
    }


def make_openai_tool_call_response(model="gpt-4", query="reply with pong"):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bench_tool_1",
                            "type": "function",
                            "function": {
                                "name": "tool_echo",
                                "arguments": json.dumps({"query": query}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
        },
    }


def extract_user_query(payload):
    for message in reversed(payload.get("messages", [])):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return "reply with pong"


def extract_tool_result(payload):
    for message in reversed(payload.get("messages", [])):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            content_json = json.loads(content)
        except json.JSONDecodeError:
            return content
        if isinstance(content_json, dict):
            result = content_json.get("result")
            if isinstance(result, str):
                return result
        return content
    return "pong"


SSE_CHUNKS = [
    "Hello",
    "! I'm",
    " a mock",
    " LLM",
    " response",
    " for testing",
    " the gateway",
    " plugins.",
]


def split_route_prefix(path):
    clean_path = urlsplit(path).path
    for prefix in ("/alpha", "/beta", "/semantic-cache-hit"):
        if clean_path == prefix:
            return prefix[1:], "/"
        if clean_path.startswith(prefix + "/"):
            return prefix[1:], clean_path[len(prefix) :]
    return None, clean_path


class LLMHandler(BaseHTTPRequestHandler):
    latency = 0.0
    log_requests = True
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        model = payload.get("model", "gpt-4")
        stream = payload.get("stream", False)
        prompt_cache_enabled = bool(
            payload.get("prompt_cache_key")
            or payload.get("prompt_cache_retention")
        )

        route_prefix, effective_path = split_route_prefix(self.path)

        if effective_path.startswith("/tool/echo"):
            self._handle_tool_echo(payload)
            return

        # Also check query param ?stream=true
        if "stream=true" in (self.path or ""):
            stream = True

        if self.log_requests:
            auth = self.headers.get("Authorization", "")
            print(
                f"  [{self.path}] route={route_prefix or 'default'} model={model} stream={stream} prompt_cache={prompt_cache_enabled} auth={auth[:20]}..."
            )

        if route_prefix == "semantic-cache-hit":
            self._handle_semantic_cache_hit(model)
            return
        if route_prefix == "alpha":
            self._handle_openai(
                model,
                prompt_cache_enabled,
                content="Alpha handled the request.",
            )
            return
        if route_prefix == "beta":
            self._handle_openai(
                model,
                prompt_cache_enabled,
                content="Beta handled the request.",
            )
            return
        if stream:
            self._handle_stream(model)
        elif payload.get("benchmark_direct_tool_round_trip"):
            self._handle_direct_tool_round_trip(model, payload)
        elif effective_path.startswith("/v1/messages"):
            self._handle_anthropic(model)
        elif any(message.get("role") == "tool" for message in payload.get("messages", [])):
            self._handle_openai_tool_followup(model, payload)
        elif payload.get("tools"):
            self._handle_openai_tool_call(model, payload)
        else:
            self._handle_openai(model, prompt_cache_enabled)

    def _handle_openai(self, model, prompt_cache_enabled, content=None):
        resp = json.dumps(
            make_openai_response(
                model,
                prompt_cache_enabled,
                content=content,
            )
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _handle_openai_tool_call(self, model, payload):
        resp = json.dumps(
            make_openai_tool_call_response(model, extract_user_query(payload))
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _handle_openai_tool_followup(self, model, payload):
        result = extract_tool_result(payload)
        resp = json.dumps(
            make_openai_response(
                model,
                prompt_cache_enabled=False,
                content=f"Tool returned {result}",
            )
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _handle_anthropic(self, model):
        resp = json.dumps(make_anthropic_response(model)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _handle_tool_echo(self, payload):
        query = payload.get("arguments", {}).get("query", "reply with pong")
        result = {"result": "pong", "query": query}
        resp = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _handle_direct_tool_round_trip(self, model, payload):
        url = f"http://127.0.0.1:{self.server.server_port}/tool/echo"
        tool_request = Request(
            url,
            data=json.dumps({"arguments": {"query": extract_user_query(payload)}}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(tool_request, timeout=2.0) as resp:
            tool_body = json.loads(resp.read().decode())
        final = json.dumps(
            make_openai_response(
                model,
                prompt_cache_enabled=False,
                content=f"Tool returned {tool_body.get('result', 'pong')}",
            )
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(final)))
        self.end_headers()
        self.wfile.write(final)

    def _handle_semantic_cache_hit(self, model):
        resp = json.dumps(
            make_openai_response(
                model,
                prompt_cache_enabled=False,
                content="Reset your password from the account settings page.",
            )
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _handle_stream(self, model):
        """SSE streaming response (OpenAI-style)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        for i, chunk in enumerate(SSE_CHUNKS):
            data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
            if self.latency > 0:
                time.sleep(self.latency)

        # Final chunk with finish_reason
        final = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, format, *args):
        # Suppress default logging, we print our own
        pass


class BenchmarkHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def main():
    parser = argparse.ArgumentParser(description="Mock LLM API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9900)
    parser.add_argument("--latency", type=float, default=0.0,
                        help="Latency per SSE chunk in seconds")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-request benchmark logging",
    )
    args = parser.parse_args()

    LLMHandler.latency = args.latency
    LLMHandler.log_requests = not args.quiet

    server = BenchmarkHTTPServer((args.host, args.port), LLMHandler)
    print(f"Mock LLM API listening on http://{args.host}:{args.port}")
    print(f"  POST /v1/chat/completions   (OpenAI)")
    print(f"  POST /v1/messages           (Anthropic)")
    print(f"  Add ?stream=true for SSE    (latency={args.latency}s)")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
