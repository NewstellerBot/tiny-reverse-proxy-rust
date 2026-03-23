#!/usr/bin/env python3
"""
HTTP/3 spike test for request-body limit behavior.

Expected behavior:
- Most oversized HTTP/3 POST requests are rejected with 413.
- After the spike, HTTP/3 GET health checks still return 200.
- Proxy RSS growth stays below a configurable threshold.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from stress_common import (
    ensure_h3_spike_client_binary,
    ensure_proxy_binary,
    has_curl_http3,
    process_rss_kb,
    start_proxy,
    start_upstream,
    terminate_process,
    wait_for_http_status,
    wait_for_tcp_port,
    write_temp_config,
)


def run_h3_post(url: str, payload_path: Path) -> tuple[str, float]:
    cmd = [
        "curl",
        "--http3-only",
        "-k",
        "-sS",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "--data-binary",
        f"@{payload_path}",
        url,
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        status = result.stdout.strip() or "000"
    else:
        status = f"ERR:{result.returncode}"
    return status, elapsed


def run_h3_get(url: str) -> str:
    cmd = [
        "curl",
        "--http3-only",
        "-k",
        "-sS",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip() or "000"
    return f"ERR:{result.returncode}"


def parse_h3_client_output(stdout: str) -> tuple[dict[str, int], float, float]:
    counts: dict[str, int] = {}
    p50_ms = math.inf
    p95_ms = math.inf

    for line in stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "STATUS" and len(parts) == 3:
            try:
                counts[parts[1]] = int(parts[2])
            except ValueError:
                pass
        elif parts[0] == "LAT_P50_MS" and len(parts) == 2:
            try:
                p50_ms = float(parts[1])
            except ValueError:
                pass
        elif parts[0] == "LAT_P95_MS" and len(parts) == 2:
            try:
                p95_ms = float(parts[1])
            except ValueError:
                pass

    return counts, p50_ms, p95_ms


def run_h3_batch_with_rust_client(
    *,
    h3_client_bin: Path,
    url: str,
    method: str,
    requests: int,
    concurrency: int,
    ca_cert_path: Path | None = None,
    insecure: bool = False,
    body_file: Path | None = None,
    timeout_ms: int = 8000,
) -> tuple[dict[str, int], float, float]:
    cmd = [
        str(h3_client_bin),
        "--url",
        url,
        "--method",
        method,
        "--requests",
        str(requests),
        "--concurrency",
        str(concurrency),
        "--server-name",
        "localhost",
        "--timeout-ms",
        str(timeout_ms),
    ]
    if insecure:
        cmd.append("--insecure")
    else:
        if ca_cert_path is None:
            raise ValueError("ca_cert_path is required unless insecure=True")
        cmd.extend(["--ca-cert", str(ca_cert_path)])
    if body_file is not None:
        cmd.extend(["--body-file", str(body_file)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "h3_spike_client failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return parse_h3_client_output(result.stdout)


def percentile_ms(values_sec: list[float], p: float) -> float:
    if not values_sec:
        return math.inf
    arr = sorted(values_sec)
    idx = min(len(arr) - 1, max(0, int(math.ceil((p / 100.0) * len(arr))) - 1))
    return arr[idx] * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP/3 oversized-body spike test")
    parser.add_argument("--proxy-port", type=int, default=18443)
    parser.add_argument("--upstream-port", type=int, default=19000)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit-bytes", type=int, default=64 * 1024)
    parser.add_argument("--oversized-bytes", type=int, default=256 * 1024)
    parser.add_argument("--post-check-gets", type=int, default=20)
    parser.add_argument("--min-413-ratio", type=float, default=0.98)
    parser.add_argument("--max-rss-growth-kb", type=int, default=128 * 1024)
    parser.add_argument("--proxy-bin", type=str, default=None)
    args = parser.parse_args()

    if args.oversized_bytes <= args.limit_bytes:
        print("FAIL: oversized-bytes must be greater than limit-bytes", file=sys.stderr)
        return 2

    proxy_bin = ensure_proxy_binary(args.proxy_bin)
    config = write_temp_config(
        "\n".join(
            [
                f"port = {args.proxy_port}",
                "tls = true",
                f"max_request_body_bytes = {args.limit_bytes}",
                "",
                "[paths]",
                f'"/*" = ["127.0.0.1:{args.upstream_port}"]',
            ]
        )
    )

    upstream_proc = None
    proxy_proc = None
    payload_file: Path | None = None
    try:
        upstream_proc = start_upstream(args.upstream_port)
        wait_for_tcp_port("127.0.0.1", args.upstream_port, timeout_sec=10)

        proxy_proc = start_proxy(proxy_bin, config)
        wait_for_tcp_port("127.0.0.1", args.proxy_port, timeout_sec=10)

        base_url = f"https://127.0.0.1:{args.proxy_port}/"
        upload_url = f"https://127.0.0.1:{args.proxy_port}/upload"

        wait_for_http_status(base_url, "200", timeout_sec=20, insecure_tls=True)

        use_curl_http3 = has_curl_http3()
        h3_client_bin: Path | None = None
        if use_curl_http3:
            wait_for_http_status(
                base_url,
                "200",
                timeout_sec=20,
                insecure_tls=True,
                http3_only=True,
            )
        else:
            h3_client_bin = ensure_h3_spike_client_binary()
            ready_counts, _, _ = run_h3_batch_with_rust_client(
                h3_client_bin=h3_client_bin,
                url=base_url,
                method="GET",
                requests=1,
                concurrency=1,
                insecure=True,
            )
            if ready_counts.get("200", 0) != 1:
                raise RuntimeError(
                    "HTTP/3 readiness check failed for rust-h3-client backend"
                )

        rss_before = process_rss_kb(proxy_proc.pid)

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_payload:
            tmp_payload.write(os.urandom(args.oversized_bytes))
            payload_file = Path(tmp_payload.name)

        lat_p50_ms = math.inf
        lat_p95_ms = math.inf
        if use_curl_http3:
            statuses: list[str] = []
            latencies: list[float] = []

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [
                    ex.submit(run_h3_post, upload_url, payload_file)
                    for _ in range(args.requests)
                ]
                for fut in as_completed(futures):
                    status, elapsed = fut.result()
                    statuses.append(status)
                    latencies.append(elapsed)

            counts = Counter(statuses)
            lat_p50_ms = percentile_ms(latencies, 50)
            lat_p95_ms = percentile_ms(latencies, 95)
        else:
            assert h3_client_bin is not None
            counts_dict, lat_p50_ms, lat_p95_ms = run_h3_batch_with_rust_client(
                h3_client_bin=h3_client_bin,
                url=upload_url,
                method="POST",
                requests=args.requests,
                concurrency=args.concurrency,
                insecure=True,
                body_file=payload_file,
            )
            counts = Counter(counts_dict)

        count_413 = counts.get("413", 0)
        ratio_413 = count_413 / args.requests if args.requests else 0.0

        if use_curl_http3:
            post_get_statuses = [run_h3_get(base_url) for _ in range(args.post_check_gets)]
            post_get_200 = sum(1 for code in post_get_statuses if code == "200")
        else:
            assert h3_client_bin is not None
            get_counts, _, _ = run_h3_batch_with_rust_client(
                h3_client_bin=h3_client_bin,
                url=base_url,
                method="GET",
                requests=args.post_check_gets,
                concurrency=min(8, args.post_check_gets),
                insecure=True,
                body_file=None,
            )
            post_get_200 = get_counts.get("200", 0)

        rss_after = process_rss_kb(proxy_proc.pid)
        rss_growth = None
        if rss_before is not None and rss_after is not None:
            rss_growth = max(0, rss_after - rss_before)

        print("\n=== HTTP/3 large-body spike summary ===")
        print(f"requests={args.requests} concurrency={args.concurrency}")
        print(f"http3_backend={'curl' if use_curl_http3 else 'rust-h3-client'}")
        print(f"status_counts={dict(sorted(counts.items()))}")
        print(f"413_ratio={ratio_413:.3f} (min={args.min_413_ratio:.3f})")
        print(
            f"latency_ms_p50={lat_p50_ms:.1f} "
            f"latency_ms_p95={lat_p95_ms:.1f}"
        )
        print(f"post_spike_h3_get_200={post_get_200}/{args.post_check_gets}")
        if rss_growth is None:
            print("rss_growth_kb=unknown")
        else:
            print(f"rss_growth_kb={rss_growth} (max={args.max_rss_growth_kb})")

        failures: list[str] = []
        if ratio_413 < args.min_413_ratio:
            failures.append(
                f"413 ratio too low: got {ratio_413:.3f}, expected at least {args.min_413_ratio:.3f}"
            )
        if post_get_200 != args.post_check_gets:
            failures.append(
                f"post-spike health failed: {post_get_200}/{args.post_check_gets} were HTTP 200"
            )
        if rss_growth is not None and rss_growth > args.max_rss_growth_kb:
            failures.append(
                f"rss growth too high: {rss_growth} KB > {args.max_rss_growth_kb} KB"
            )

        if failures:
            print("\nFAIL")
            for f in failures:
                print(f"- {f}")
            return 1

        print("\nPASS")
        return 0
    finally:
        if payload_file and payload_file.exists():
            payload_file.unlink(missing_ok=True)
        if proxy_proc is not None:
            terminate_process(proxy_proc, "proxy")
        if upstream_proc is not None:
            terminate_process(upstream_proc, "upstream")
        if config.exists():
            config.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
