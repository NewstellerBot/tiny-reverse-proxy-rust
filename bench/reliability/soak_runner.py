import argparse
import http.client
import json
import os
import queue
import random
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_TOKEN = "release-soak-bootstrap-admin"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def percentile(samples: list[float], p: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * p)))
    return ordered[index]


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress(message: str) -> None:
    print(f"[soak] {message}", file=sys.stderr, flush=True)


def http_json(method: str, url: str, payload=None, token: str | None = None) -> tuple[int, bytes]:
    body = None
    headers = {"connection": "close"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["content-type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=60) as resp:
            return resp.getcode(), resp.read()
    except error.HTTPError as exc:
        return exc.code, exc.read()
    except (error.URLError, TimeoutError, http.client.RemoteDisconnected) as exc:
        return 0, str(exc).encode("utf-8", errors="replace")


class FlakyMockHandler(BaseHTTPRequestHandler):
    sequence = Counter()
    rate_limit_every = 7
    upstream_error_every = 11

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body or b"{}")
        key = self.path
        FlakyMockHandler.sequence[key] += 1
        hit = FlakyMockHandler.sequence[key]

        if self.rate_limit_every and hit % self.rate_limit_every == 0:
            body = b"rate limited"
            self.send_response(429)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.upstream_error_every and hit % self.upstream_error_every == 0:
            body = b"upstream error"
            self.send_response(500)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/v1/chat/completions":
            body_bytes = json.dumps(
                {
                    "id": f"chatcmpl-{hit}",
                    "object": "chat.completion",
                    "model": payload.get("model", "gpt-4o"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
            return

        if self.path == "/v1/responses" and payload.get("stream"):
            body_text = (
                "event: response.output_text.delta\n"
                'data: {"delta":"ok"}\n\n'
                "event: response.completed\n"
                'data: {"id":"resp_done"}\n\n'
            )
            body_bytes = body_text.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
            return

        if self.path == "/v1/responses":
            body_bytes = json.dumps(
                {
                    "id": f"resp-{hit}",
                    "object": "response",
                    "model": payload.get("model", "gpt-4.1-mini"),
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                }
            ).encode("utf-8")
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


class HealthyMockHandler(FlakyMockHandler):
    def do_POST(self):
        FlakyMockHandler.sequence[self.path] += 1
        hit = FlakyMockHandler.sequence[self.path]
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body or b"{}")
        if self.path == "/v1/chat/completions":
            response = {
                "id": f"chatcmpl-fallback-{hit}",
                "object": "chat.completion",
                "model": payload.get("model", "gpt-4o"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        else:
            response = {
                "id": f"resp-fallback-{hit}",
                "object": "response",
                "model": payload.get("model", "gpt-4.1-mini"),
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }
        body_bytes = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)


class TunedMockHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 4096


@dataclass
class LaneStats:
    name: str
    latencies_ms: list[float] = field(default_factory=list)
    statuses: Counter = field(default_factory=Counter)
    failures: list[str] = field(default_factory=list)
    requests: int = 0

    def record(self, status: int, latency_ms: float, failure: str | None = None) -> None:
        self.requests += 1
        self.statuses[str(status)] += 1
        self.latencies_ms.append(latency_ms)
        if failure:
            self.failures.append(failure)

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "statuses": dict(self.statuses),
            "p50_ms": percentile(self.latencies_ms, 0.50),
            "p95_ms": percentile(self.latencies_ms, 0.95),
            "p99_ms": percentile(self.latencies_ms, 0.99),
            "failures": self.failures[:10],
        }


@dataclass
class MockRequestResult:
    scenario: str
    status: int
    latency_ms: float
    failure: str | None


class MockLoadGenerator:
    def __init__(
        self,
        base_url: str,
        token: str,
        fixtures: dict,
        target_rps: float,
        concurrency: int,
        max_inflight: int,
    ) -> None:
        self.base_url = base_url
        self.token = token
        self.fixtures = fixtures
        self.target_rps = target_rps
        self.concurrency = concurrency
        self.max_inflight = max_inflight
        self.scenarios = tuple(fixtures.keys())
        self.task_queue: queue.Queue[str] = queue.Queue(maxsize=max_inflight)
        self.result_queue: queue.SimpleQueue[MockRequestResult] = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.active_requests = 0
        self.max_inflight_observed = 0
        self.submitted = 0
        self.completed = 0
        self.backpressure_events = 0
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []

    def start(self) -> None:
        for index in range(self.concurrency):
            worker = threading.Thread(target=self._worker, name=f"mock-soak-{index}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def _update_max_inflight(self) -> None:
        current = self.active_requests + self.task_queue.qsize()
        if current > self.max_inflight_observed:
            self.max_inflight_observed = current

    def _worker(self) -> None:
        while True:
            if self.stop_event.is_set() and self.task_queue.empty():
                return
            try:
                scenario = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._lock:
                self.active_requests += 1
                self._update_max_inflight()

            try:
                payload = self.fixtures[scenario]
                started = time.perf_counter()
                status, body = http_json(
                    "POST",
                    f"{self.base_url}{payload['path']}",
                    payload["body"],
                    token=self.token,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                failure = None if status == 200 else (
                    f"{scenario}:{status}:{body.decode('utf-8', errors='replace')[:120]}"
                )
                self.result_queue.put(MockRequestResult(scenario, status, latency_ms, failure))
            finally:
                with self._lock:
                    self.active_requests -= 1
                self.task_queue.task_done()

    def dispatch_until(self, elapsed_secs: float) -> None:
        desired_submissions = int(elapsed_secs * self.target_rps)
        while self.submitted < desired_submissions:
            scenario = self.scenarios[self.submitted % len(self.scenarios)]
            try:
                self.task_queue.put_nowait(scenario)
            except queue.Full:
                self.backpressure_events += 1
                return
            self.submitted += 1
            with self._lock:
                self._update_max_inflight()

    def drain_results(self) -> list[MockRequestResult]:
        drained = []
        while True:
            try:
                drained.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        self.completed += len(drained)
        return drained

    def inflight(self) -> int:
        with self._lock:
            return self.active_requests + self.task_queue.qsize()

    def shutdown(self, timeout_secs: float = 10.0) -> None:
        self.stop_event.set()
        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            self.drain_results()
            if self.inflight() == 0:
                break
            time.sleep(0.05)
        for worker in self._workers:
            worker.join(timeout=1.0)


def wait_ready(base_url: str, timeout_secs: int = 30) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            with request.urlopen(f"{base_url}/_trp/readyz", timeout=2) as resp:
                if resp.getcode() == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("proxy did not become ready")


def wait_management_ready(management_port: int, timeout_secs: int = 30) -> None:
    deadline = time.time() + timeout_secs
    url = f"http://127.0.0.1:{management_port}/api/v1/status"
    while time.time() < deadline:
        try:
            req = request.Request(
                url,
                headers={"authorization": f"Bearer {BOOTSTRAP_TOKEN}"},
                method="GET",
            )
            with request.urlopen(req, timeout=2) as resp:
                if resp.getcode() == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("management API did not become ready")


def collect_process_output(process: subprocess.Popen) -> str:
    if process.stdout is None:
        return ""
    try:
        return process.stdout.read() or ""
    except Exception:
        return ""


def create_key(base_url: str, management_port: int, provider_name: str, key_name: str) -> str:
    mgmt = f"http://127.0.0.1:{management_port}"
    status, _ = http_json(
        "POST",
        f"{mgmt}/api/v1/projects",
        {"project_id": "project-a", "name": "Project A"},
        token=BOOTSTRAP_TOKEN,
    )
    if status not in (201, 409):
        raise RuntimeError(f"project create failed: {status}")
    status, body = http_json(
        "POST",
        f"{mgmt}/api/v1/keys",
        {"project_id": "project-a", "name": key_name, "provider_name": provider_name},
        token=BOOTSTRAP_TOKEN,
    )
    if status != 201:
        raise RuntimeError(f"key create failed: {status} {body.decode('utf-8', errors='replace')}")
    return json.loads(body)["key"]


def scrape_metrics(metrics_port: int) -> str:
    with request.urlopen(f"http://127.0.0.1:{metrics_port}/metrics", timeout=5) as resp:
        return resp.read().decode("utf-8", errors="replace")


def sample_process(pid: int) -> dict:
    rss = None
    cpu = None
    try:
        rss = (
            subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
            .strip()
        )
    except Exception:
        pass
    try:
        cpu = (
            subprocess.check_output(["ps", "-o", "%cpu=", "-p", str(pid)], text=True)
            .strip()
        )
    except Exception:
        pass
    return {"rss_kb": rss, "cpu_percent": cpu}


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Run the release-candidate soak in mixed, live-only, or mock-only mode."
    )
    parser.add_argument("--duration-secs", type=int, default=300)
    parser.add_argument("--binary", default=str(REPO_ROOT / "target" / "debug" / "tiny-reverse-proxy"))
    parser.add_argument("--progress-interval-secs", type=int, default=15)
    parser.add_argument(
        "--mock-target-rps",
        type=float,
        default=6.0,
        help="Target aggregate request rate for the mock lane.",
    )
    parser.add_argument(
        "--mock-concurrency",
        type=int,
        default=128,
        help="Number of worker threads used for the mock lane.",
    )
    parser.add_argument(
        "--mock-max-inflight",
        type=int,
        default=4096,
        help="Maximum outstanding mock requests waiting or in flight at once.",
    )
    parser.add_argument(
        "--proxy-max-inflight-requests",
        type=int,
        default=256,
        help="Value to write into [reliability].max_inflight_requests for the proxy under test.",
    )
    parser.add_argument(
        "--proxy-brownout-inflight-requests",
        type=int,
        default=192,
        help="Value to write into [reliability].brownout_inflight_requests for the proxy under test.",
    )
    parser.add_argument(
        "--mock-rate-limit-every",
        type=int,
        default=7,
        help="Inject a 429 every N hits on the flaky mock primary. Use 0 to disable.",
    )
    parser.add_argument(
        "--mock-upstream-error-every",
        type=int,
        default=11,
        help="Inject a 500 every N hits on the flaky mock primary. Use 0 to disable.",
    )
    parser.add_argument(
        "--mode",
        choices=("mixed", "live-only", "mock-only"),
        default="mixed",
        help="Select whether to generate mixed traffic, live-provider-only traffic, or mock-only traffic.",
    )
    args = parser.parse_args()
    live_api_key = os.environ.get("OPENAI_API_KEY")
    wants_mock = args.mode in ("mixed", "mock-only")
    wants_live = args.mode in ("mixed", "live-only") and bool(live_api_key)

    if args.mode == "live-only" and not live_api_key:
        parser.error("--mode live-only requires OPENAI_API_KEY in the environment")
    if wants_mock and args.mock_target_rps <= 0:
        parser.error("--mock-target-rps must be greater than 0 when the mock lane is enabled")
    if wants_mock and args.mock_concurrency <= 0:
        parser.error("--mock-concurrency must be greater than 0 when the mock lane is enabled")
    if wants_mock and args.mock_max_inflight <= 0:
        parser.error("--mock-max-inflight must be greater than 0 when the mock lane is enabled")
    if args.proxy_max_inflight_requests <= 0:
        parser.error("--proxy-max-inflight-requests must be greater than 0")
    if args.proxy_brownout_inflight_requests <= 0:
        parser.error("--proxy-brownout-inflight-requests must be greater than 0")
    if args.proxy_brownout_inflight_requests >= args.proxy_max_inflight_requests:
        parser.error("--proxy-brownout-inflight-requests must be less than --proxy-max-inflight-requests")
    if args.mock_rate_limit_every < 0:
        parser.error("--mock-rate-limit-every must be 0 or greater")
    if args.mock_upstream_error_every < 0:
        parser.error("--mock-upstream-error-every must be 0 or greater")

    default_binary = REPO_ROOT / "target" / "debug" / "tiny-reverse-proxy"
    binary = Path(args.binary)
    if binary == default_binary:
        progress("building gateway-enabled proxy binary")
        subprocess.run(
            ["cargo", "build", "-p", "tiny-reverse-proxy", "--features", "plugin-llm-gateway"],
            cwd=REPO_ROOT,
            check=True,
        )
    elif not binary.exists():
        raise FileNotFoundError(f"binary not found: {binary}")

    workspace = Path(tempfile.mkdtemp(prefix="trp-soak-", dir=REPO_ROOT / "bench" / "reliability"))
    primary_port = free_port()
    fallback_port = free_port()
    proxy_port = free_port()
    management_port = free_port()
    metrics_port = free_port()

    FlakyMockHandler.sequence = Counter()
    FlakyMockHandler.rate_limit_every = args.mock_rate_limit_every
    FlakyMockHandler.upstream_error_every = args.mock_upstream_error_every

    primary = TunedMockHTTPServer(("127.0.0.1", primary_port), FlakyMockHandler)
    fallback = TunedMockHTTPServer(("127.0.0.1", fallback_port), HealthyMockHandler)
    primary_thread = threading.Thread(target=primary.serve_forever, daemon=True)
    fallback_thread = threading.Thread(target=fallback.serve_forever, daemon=True)
    primary_thread.start()
    fallback_thread.start()

    template = (
        REPO_ROOT / "bench" / "reliability" / "fixtures" / "soak_proxy.toml.tpl"
    ).read_text()
    config_path = workspace / "soak.toml"
    sqlite_path = workspace / "soak.db"
    live_provider_block = ""
    if wants_live and live_api_key:
        live_provider_block = """
[[providers]]
name = "openai-live"
api_key = "$OPENAI_API_KEY"
base_url = "https://api.openai.com"
models = ["gpt-4o", "gpt-4.1-mini"]
api_key_header = "authorization"
family = "openai"

[providers.surfaces]
tools = "openai"
responses = "openai_compatible"

[[plugins.config.providers]]
name = "openai-live"
pattern = "https://api.openai.com"
""".strip()
    config_path.write_text(
        template.format(
            port=proxy_port,
            management_api_port=management_port,
            metrics_port=metrics_port,
            sqlite_path=sqlite_path,
            mock_primary_port=primary_port,
            mock_fallback_port=fallback_port,
            proxy_max_inflight_requests=args.proxy_max_inflight_requests,
            proxy_brownout_inflight_requests=args.proxy_brownout_inflight_requests,
            live_provider_block=live_provider_block,
        )
    )
    progress(
        f"prepared soak workspace at {workspace} "
        f"(duration={format_duration(args.duration_secs)}, mode={args.mode}, "
        f"mock_lane={'on' if wants_mock else 'off'}, live_lane={'on' if wants_live else 'off'}, "
        f"mock_target_rps={args.mock_target_rps if wants_mock else 'n/a'}, "
        f"mock_concurrency={args.mock_concurrency if wants_mock else 'n/a'}, "
        f"proxy_max_inflight={args.proxy_max_inflight_requests}, "
        f"proxy_brownout={args.proxy_brownout_inflight_requests}, "
        f"mock_429_every={args.mock_rate_limit_every}, mock_500_every={args.mock_upstream_error_every})"
    )

    env = os.environ.copy()
    env["TRP_BOOTSTRAP_ADMIN_TOKEN"] = BOOTSTRAP_TOKEN
    process = subprocess.Popen(
        [str(binary), str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proxy_log_tail: deque[str] = deque(maxlen=40)

    def pump_logs() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            proxy_log_tail.append(line.rstrip())

    threading.Thread(target=pump_logs, daemon=True).start()

    try:
        base_url = f"http://127.0.0.1:{proxy_port}"
        progress("waiting for proxy readiness")
        wait_ready(base_url)
        progress("waiting for management API readiness")
        wait_management_ready(management_port)
        mock_key = None
        if wants_mock:
            progress("creating mock soak key")
            mock_key = create_key(base_url, management_port, "mock-primary", "mock-soak-key")
        live_key = None
        if wants_live:
            progress("creating live soak key")
            live_key = create_key(base_url, management_port, "openai-live", "live-soak-key")
        progress("startup complete; beginning soak loop")

        fixtures = {}
        if wants_mock:
            fixtures = {
                "chat": json.loads(
                    (REPO_ROOT / "bench" / "reliability" / "fixtures" / "mock_chat.json").read_text()
                ),
                "responses": json.loads(
                    (REPO_ROOT / "bench" / "reliability" / "fixtures" / "mock_responses.json").read_text()
                ),
                "responses_stream": json.loads(
                    (REPO_ROOT / "bench" / "reliability" / "fixtures" / "mock_responses_stream.json").read_text()
                ),
            }

        mock_stats = LaneStats("mock")
        live_stats = LaneStats("live")
        process_samples = []
        deadline = time.time() + args.duration_secs
        started_at = time.perf_counter()
        next_live = time.time()
        next_live_stream = time.time()
        next_progress = time.time() + max(1, args.progress_interval_secs)
        catastrophic_failure = False
        status_zero_reported = False
        mock_load = None
        if wants_mock:
            mock_load = MockLoadGenerator(
                base_url=base_url,
                token=mock_key,
                fixtures=fixtures,
                target_rps=args.mock_target_rps,
                concurrency=args.mock_concurrency,
                max_inflight=args.mock_max_inflight,
            )
            progress(
                f"starting mock load generator at target {args.mock_target_rps:.1f} RPS "
                f"with concurrency={args.mock_concurrency} max_inflight={args.mock_max_inflight}"
            )
            mock_load.start()

        while time.time() < deadline:
            if process.poll() is not None:
                catastrophic_failure = True
                break
            if mock_load is not None:
                mock_load.dispatch_until(time.perf_counter() - started_at)
                for result in mock_load.drain_results():
                    mock_stats.record(result.status, result.latency_ms, result.failure)
                    if result.status == 0 and not status_zero_reported:
                        status_zero_reported = True
                        progress(
                            "observed status 0 in mock lane; proxy_log_tail will be included in the final summary"
                        )

            now = time.time()
            if live_key and now >= next_live:
                started = time.perf_counter()
                status, body = http_json(
                    "POST",
                    f"{base_url}/v1/responses",
                    {"model": "gpt-4.1-mini", "input": "health check"},
                    token=live_key,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                failure = None if status == 200 else (
                    f"responses:{status}:{body.decode('utf-8', errors='replace')[:120]}"
                )
                live_stats.record(status, latency_ms, failure)
                if status == 0 and not status_zero_reported:
                    status_zero_reported = True
                    progress("observed status 0 in live lane; proxy_log_tail will be included in the final summary")
                next_live = now + 60
            if live_key and now >= next_live_stream:
                started = time.perf_counter()
                status, body = http_json(
                    "POST",
                    f"{base_url}/v1/responses",
                    {"model": "gpt-4.1-mini", "input": "stream health check", "stream": True},
                    token=live_key,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                failure = None if status == 200 else (
                    f"responses_stream:{status}:{body.decode('utf-8', errors='replace')[:120]}"
                )
                live_stats.record(status, latency_ms, failure)
                if status == 0 and not status_zero_reported:
                    status_zero_reported = True
                    progress("observed status 0 in live lane; proxy_log_tail will be included in the final summary")
                next_live_stream = now + 120

            process_samples.append(sample_process(process.pid))
            if now >= next_progress:
                latest_sample = process_samples[-1] if process_samples else {"rss_kb": None, "cpu_percent": None}
                mock_statuses = dict(mock_stats.statuses) if wants_mock else {"skipped": True}
                live_statuses = dict(live_stats.statuses) if wants_live else {"skipped": True}
                elapsed_secs = max(0.001, time.perf_counter() - started_at)
                mock_achieved_rps = mock_stats.requests / elapsed_secs if wants_mock else None
                progress(
                    "elapsed="
                    f"{format_duration(args.duration_secs - max(0, deadline - now))}"
                    f"/{format_duration(args.duration_secs)} "
                    f"remaining={format_duration(max(0, deadline - now))} "
                    f"mock_requests={mock_stats.requests if wants_mock else 'skipped'} mock_statuses={mock_statuses} "
                    f"mock_rps={f'{mock_achieved_rps:.1f}/{args.mock_target_rps:.1f}' if wants_mock else 'skipped'} "
                    f"mock_inflight={mock_load.inflight() if mock_load is not None else 'skipped'} "
                    f"live_requests={live_stats.requests if wants_live else 'skipped'} live_statuses={live_statuses} "
                    f"rss_kb={latest_sample.get('rss_kb')} cpu={latest_sample.get('cpu_percent')} "
                    f"proxy_exit={process.poll()}"
                )
                next_progress = now + max(1, args.progress_interval_secs)
            time.sleep(0.01)

        if mock_load is not None:
            progress("draining mock inflight requests")
            mock_load.shutdown()
            for result in mock_load.drain_results():
                mock_stats.record(result.status, result.latency_ms, result.failure)
                if result.status == 0 and not status_zero_reported:
                    status_zero_reported = True
                    progress("observed status 0 in mock lane during drain; proxy_log_tail will be included in the final summary")

        metrics = scrape_metrics(metrics_port)
        mock_summary = mock_stats.summary() if wants_mock else {"skipped": True, "reason": "mode=live-only"}
        if wants_mock:
            mock_summary["target_rps"] = args.mock_target_rps
            mock_summary["achieved_rps"] = mock_stats.requests / max(1, args.duration_secs)
            mock_summary["concurrency"] = args.mock_concurrency
            mock_summary["max_inflight"] = args.mock_max_inflight
            mock_summary["max_inflight_observed"] = mock_load.max_inflight_observed if mock_load is not None else 0
            mock_summary["backpressure_events"] = mock_load.backpressure_events if mock_load is not None else 0
            mock_summary["proxy_max_inflight_requests"] = args.proxy_max_inflight_requests
            mock_summary["proxy_brownout_inflight_requests"] = args.proxy_brownout_inflight_requests
            mock_summary["mock_rate_limit_every"] = args.mock_rate_limit_every
            mock_summary["mock_upstream_error_every"] = args.mock_upstream_error_every
        summary = {
            "mode": args.mode,
            "duration_secs": args.duration_secs,
            "mock_lane": mock_summary,
            "live_lane": live_stats.summary()
            if wants_live
            else {
                "skipped": True,
                "reason": "mode=mock-only" if args.mode == "mock-only" else "OPENAI_API_KEY not set",
            },
            "process_samples": process_samples[-10:],
            "proxy_exit_code": process.poll(),
            "metrics_excerpt": "\n".join(
                line
                for line in metrics.splitlines()
                if "proxy_retry_" in line
                or "proxy_brownout_" in line
                or "proxy_admission_" in line
                or "llm_provider_cooldowns_total" in line
            ),
        }
        if mock_stats.statuses.get("0", 0) > 0 or live_stats.statuses.get("0", 0) > 0:
            summary["proxy_log_tail"] = list(proxy_log_tail)
        progress(
            f"completed soak: mock_statuses={dict(mock_stats.statuses) if wants_mock else {'skipped': True}} "
            f"live_statuses={dict(live_stats.statuses) if wants_live else {'skipped': True}}"
        )
        print(json.dumps(summary, indent=2))

        if wants_live and live_stats.requests == 0:
            return 1
        if wants_mock and mock_stats.requests == 0:
            return 1
        if catastrophic_failure or process.poll() not in (None, 0):
            return 1
        if wants_mock and mock_stats.statuses.get("200", 0) == 0:
            return 1
        return 0
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        elif process.returncode not in (0, None):
            logs = "\n".join(proxy_log_tail) or collect_process_output(process)
            if logs:
                print("proxy startup logs:", file=sys.stderr)
                print(logs, file=sys.stderr)
        primary.shutdown()
        fallback.shutdown()


if __name__ == "__main__":
    raise SystemExit(run())
