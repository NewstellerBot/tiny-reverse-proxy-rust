import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE_URL = "postgres://postgres:postgres@127.0.0.1:55432/trp_multinode"
BOOTSTRAP_TOKEN = "multinode-bootstrap-admin"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_json(method: str, url: str, payload=None, token: str | None = None) -> tuple[int, dict]:
    body = None
    headers = {"content-type": "application/json", "connection": "close"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            return resp.getcode(), json.loads(raw or b"{}")
    except error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            parsed = {"raw": raw.decode("utf-8", errors="replace")}
        return exc.code, parsed


class MockLlmHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            payload = json.loads(body or b"{}")
            model = payload.get("model", "unknown")
            response = {
                "id": "chatcmpl-multinode",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
            body_bytes = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


@dataclass
class Node:
    name: str
    port: int
    management_port: int
    config_path: Path
    process: subprocess.Popen

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def management_url(self) -> str:
        return f"http://127.0.0.1:{self.management_port}"


def wait_ready(node: Node, timeout_secs: int = 30) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            with request.urlopen(f"{node.base_url}/_trp/readyz", timeout=2) as resp:
                if resp.getcode() == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"{node.name} did not become ready")


def wait_management_ready(node: Node, timeout_secs: int = 30) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            req = request.Request(
                f"{node.management_url}/api/v1/status",
                headers={"authorization": f"Bearer {BOOTSTRAP_TOKEN}"},
                method="GET",
            )
            with request.urlopen(req, timeout=2) as resp:
                if resp.getcode() == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"{node.name} management API did not become ready")


def start_node(binary: Path, config_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["TRP_BOOTSTRAP_ADMIN_TOKEN"] = BOOTSTRAP_TOKEN
    return subprocess.Popen(
        [str(binary), str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_node(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def load_balanced_request(nodes: list[Node], api_key: str) -> int:
    for node in nodes:
        try:
            with request.urlopen(f"{node.base_url}/_trp/readyz", timeout=2) as resp:
                if resp.getcode() != 200:
                    continue
        except Exception:
            continue
        status, _ = http_json(
            "POST",
            f"{node.base_url}/v1/chat/completions",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
            token=api_key,
        )
        return status
    raise RuntimeError("no ready nodes available")


def create_virtual_key(node: Node) -> str:
    status, _ = http_json(
        "POST",
        f"{node.management_url}/api/v1/projects",
        {"project_id": "project-a", "name": "Project A"},
        token=BOOTSTRAP_TOKEN,
    )
    if status not in (201, 409):
        raise RuntimeError(f"project create failed on {node.name}: {status}")

    status, body = http_json(
        "POST",
        f"{node.management_url}/api/v1/keys",
        {"project_id": "project-a", "name": "multi-node-key", "provider_name": "openai"},
        token=BOOTSTRAP_TOKEN,
    )
    if status != 201:
        raise RuntimeError(f"key create failed on {node.name}: {status} {body}")
    return body["key"]


def export_project_config(node: Node) -> dict:
    status, body = http_json(
        "GET",
        f"{node.management_url}/api/v1/projects/project-a/config/export",
        token=BOOTSTRAP_TOKEN,
    )
    if status != 200:
        raise RuntimeError(f"config export failed on {node.name}: {status} {body}")
    return body


def apply_policy_change(node: Node, budget_limit: float) -> None:
    status, body = http_json(
        "PUT",
        f"{node.management_url}/api/v1/projects/project-a/policy",
        {"budget_limit": budget_limit, "fallback_order": ["openai"]},
        token=BOOTSTRAP_TOKEN,
    )
    if status != 200:
        raise RuntimeError(f"policy update failed on {node.name}: {status} {body}")


def apply_project_snapshot(node: Node, snapshot: dict) -> None:
    status, body = http_json(
        "PUT",
        f"{node.management_url}/api/v1/projects/project-a/config",
        snapshot,
        token=BOOTSTRAP_TOKEN,
    )
    if status != 200:
        raise RuntimeError(f"config apply failed on {node.name}: {status} {body}")


def assert_policy_budget(node: Node, expected: float) -> None:
    status, body = http_json(
        "GET",
        f"{node.management_url}/api/v1/projects/project-a/policy",
        token=BOOTSTRAP_TOKEN,
    )
    if status != 200:
        raise RuntimeError(f"policy read failed on {node.name}: {status} {body}")
    if body.get("budget_limit") != expected:
        raise RuntimeError(
            f"{node.name} policy budget mismatch: expected {expected}, got {body.get('budget_limit')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the multi-node correctness harness against three local proxy nodes."
    )
    parser.add_argument("--binary", default=str(REPO_ROOT / "target" / "debug" / "tiny-reverse-proxy"))
    parser.add_argument("--store-url", default=DEFAULT_STORE_URL)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()

    default_binary = REPO_ROOT / "target" / "debug" / "tiny-reverse-proxy"
    binary = Path(args.binary)
    if binary == default_binary:
        subprocess.run(
            ["cargo", "build", "-p", "tiny-reverse-proxy", "--features", "plugin-llm-gateway"],
            cwd=REPO_ROOT,
            check=True,
        )
    elif not binary.exists():
        raise FileNotFoundError(f"binary not found: {binary}")

    upstream_port = free_port()
    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), MockLlmHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    template = (
        REPO_ROOT / "tests" / "reliability" / "multinode" / "templates" / "node.toml.tpl"
    ).read_text()
    workspace = Path(tempfile.mkdtemp(prefix="trp-multinode-", dir=REPO_ROOT / "tests" / "reliability"))

    nodes: list[Node] = []
    try:
        for index in range(3):
            port = free_port()
            management_port = free_port()
            config_path = workspace / f"node-{index+1}.toml"
            config_path.write_text(
                template.format(
                    port=port,
                    management_api_port=management_port,
                    store_url=args.store_url,
                    primary_upstream_port=upstream_port,
                )
            )
            process = start_node(binary, config_path)
            node = Node(
                name=f"node-{index+1}",
                port=port,
                management_port=management_port,
                config_path=config_path,
                process=process,
            )
            nodes.append(node)

        for node in nodes:
            wait_ready(node)
            wait_management_ready(node)

        api_key = create_virtual_key(nodes[0])

        # Restart nodes B and C so they reload the authoritative shared-store state.
        for node in nodes[1:]:
            stop_node(node.process)
            node.process = start_node(binary, node.config_path)
            wait_ready(node)
            wait_management_ready(node)

        for node in nodes:
            status = load_balanced_request([node], api_key)
            if status != 200:
                raise RuntimeError(f"{node.name} failed the shared-key request: {status}")

        snapshot = export_project_config(nodes[0])
        apply_policy_change(nodes[0], 25.0)
        stop_node(nodes[1].process)
        nodes[1].process = start_node(binary, nodes[1].config_path)
        wait_ready(nodes[1])
        apply_project_snapshot(nodes[0], snapshot)

        for node in nodes:
            stop_node(node.process)
            node.process = start_node(binary, node.config_path)
            wait_ready(node)
            wait_management_ready(node)
            assert_policy_budget(node, snapshot["policy"]["budget_limit"])

        # Rolling restart with readiness-aware request routing.
        for index, node in enumerate(nodes):
            status = load_balanced_request(nodes, api_key)
            if status != 200:
                raise RuntimeError(f"pre-restart request failed before restarting {node.name}: {status}")
            stop_node(node.process)
            time.sleep(1.0)
            status = load_balanced_request([n for n in nodes if n is not node], api_key)
            if status != 200:
                raise RuntimeError(f"traffic failed while {node.name} was down: {status}")
            node.process = start_node(binary, node.config_path)
            wait_ready(node)
            wait_management_ready(node)
            status = load_balanced_request(nodes, api_key)
            if status != 200:
                raise RuntimeError(f"post-restart request failed after restarting {node.name}: {status}")

        summary = {
            "result": "ok",
            "nodes": [node.name for node in nodes],
            "store_url": args.store_url,
            "workspace": str(workspace),
        }
        print(json.dumps(summary, indent=2))

        if args.keep_running:
            print("Harness complete; keeping processes running for inspection.")
            while True:
                time.sleep(60)
        return 0
    finally:
        if not args.keep_running:
            for node in nodes:
                stop_node(node.process)
            upstream.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
