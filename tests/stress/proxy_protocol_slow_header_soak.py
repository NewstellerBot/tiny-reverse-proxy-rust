#!/usr/bin/env python3
"""
Soak test for PROXY protocol slow-header pressure.

Expected behavior:
- Slow/incomplete PROXY headers do not starve valid traffic.
- Valid PROXY+HTTP requests continue to succeed during and after load.
- During-load latency remains within an acceptable p95 threshold.
"""

from __future__ import annotations

import argparse
import math
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from stress_common import (
    ensure_proxy_binary,
    start_proxy,
    start_upstream,
    terminate_process,
    wait_for_tcp_port,
    write_temp_config,
)


def percentile_ms(values_sec: list[float], p: float) -> float:
    if not values_sec:
        return math.inf
    arr = sorted(values_sec)
    idx = min(len(arr) - 1, max(0, int(math.ceil((p / 100.0) * len(arr))) - 1))
    return arr[idx] * 1000.0


def run_valid_proxy_http_request(
    host: str,
    port: int,
    timeout_sec: float,
) -> tuple[bool, float, str]:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_sec) as sock:
            sock.settimeout(timeout_sec)
            request = (
                b"PROXY TCP4 203.0.113.1 198.51.100.2 42300 443\r\n"
                b"GET / HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Connection: close\r\n"
                b"\r\n"
            )
            sock.sendall(request)

            data = b""
            while b"\r\n" not in data and len(data) < 4096:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk

        elapsed = time.perf_counter() - start
        first_line = data.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
        ok = first_line.startswith("HTTP/1.1 200") or first_line.startswith("HTTP/1.0 200")
        return ok, elapsed, first_line or "<empty>"
    except Exception as exc:  # broad: this is a stress harness
        elapsed = time.perf_counter() - start
        return False, elapsed, f"ERR:{exc}"


def run_slow_proxy_connection(host: str, port: int, hold_sec: float, timeout_sec: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec) as sock:
            sock.settimeout(timeout_sec)
            sock.sendall(b"PROXY TCP4 203.0.113.1")
            time.sleep(hold_sec)
            try:
                sock.sendall(b" 198.51")
            except OSError:
                pass
        return True
    except OSError:
        return False


def run_valid_batch(
    host: str,
    port: int,
    num_requests: int,
    concurrency: int,
    timeout_sec: float,
) -> list[tuple[bool, float, str]]:
    results: list[tuple[bool, float, str]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(run_valid_proxy_http_request, host, port, timeout_sec)
            for _ in range(num_requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def success_ratio(results: list[tuple[bool, float, str]]) -> float:
    if not results:
        return 0.0
    return sum(1 for ok, _, _ in results if ok) / len(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="PROXY slow-header soak test")
    parser.add_argument("--proxy-port", type=int, default=18080)
    parser.add_argument("--upstream-port", type=int, default=19001)
    parser.add_argument("--header-timeout-sec", type=int, default=1)
    parser.add_argument("--attack-connections", type=int, default=240)
    parser.add_argument("--attack-concurrency", type=int, default=80)
    parser.add_argument("--malicious-hold-ms", type=int, default=1300)
    parser.add_argument("--valid-during", type=int, default=120)
    parser.add_argument("--valid-after", type=int, default=40)
    parser.add_argument("--valid-concurrency", type=int, default=20)
    parser.add_argument("--valid-timeout-sec", type=float, default=2.5)
    parser.add_argument("--min-during-success-ratio", type=float, default=0.90)
    parser.add_argument("--min-after-success-ratio", type=float, default=1.0)
    parser.add_argument("--max-during-p95-ms", type=float, default=1200.0)
    parser.add_argument("--proxy-bin", type=str, default=None)
    args = parser.parse_args()

    proxy_bin = ensure_proxy_binary(args.proxy_bin)

    config = write_temp_config(
        "\n".join(
            [
                f"port = {args.proxy_port}",
                "proxy_protocol = true",
                f"header_read_timeout_secs = {args.header_timeout_sec}",
                "",
                "[paths]",
                f'"/*" = ["127.0.0.1:{args.upstream_port}"]',
            ]
        )
    )

    upstream_proc = None
    proxy_proc = None
    try:
        upstream_proc = start_upstream(args.upstream_port)
        wait_for_tcp_port("127.0.0.1", args.upstream_port, timeout_sec=10)

        proxy_proc = start_proxy(proxy_bin, config)
        wait_for_tcp_port("127.0.0.1", args.proxy_port, timeout_sec=10)

        # Baseline sanity check
        baseline_ok, _, baseline_status = run_valid_proxy_http_request(
            "127.0.0.1", args.proxy_port, args.valid_timeout_sec
        )
        if not baseline_ok:
            print(f"FAIL: baseline request did not return 200 ({baseline_status})")
            return 1

        hold_sec = args.malicious_hold_ms / 1000.0

        attack_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.attack_concurrency) as attack_pool:
            attack_futures = [
                attack_pool.submit(
                    run_slow_proxy_connection,
                    "127.0.0.1",
                    args.proxy_port,
                    hold_sec,
                    args.valid_timeout_sec,
                )
                for _ in range(args.attack_connections)
            ]

            # Run legitimate traffic while attack sockets are still being held.
            during_results = run_valid_batch(
                "127.0.0.1",
                args.proxy_port,
                args.valid_during,
                args.valid_concurrency,
                args.valid_timeout_sec,
            )

            attack_done = sum(1 for fut in as_completed(attack_futures) if fut.result())

        attack_elapsed = time.perf_counter() - attack_started

        if proxy_proc.poll() is not None:
            print("FAIL: proxy process exited during slow-header soak")
            return 1

        after_results = run_valid_batch(
            "127.0.0.1",
            args.proxy_port,
            args.valid_after,
            args.valid_concurrency,
            args.valid_timeout_sec,
        )

        during_ratio = success_ratio(during_results)
        after_ratio = success_ratio(after_results)
        during_latencies = [lat for ok, lat, _ in during_results if ok]
        during_p95_ms = percentile_ms(during_latencies, 95)

        print("\n=== PROXY slow-header soak summary ===")
        print(
            f"attack_connections={args.attack_connections} "
            f"completed={attack_done} duration_s={attack_elapsed:.2f}"
        )
        print(
            f"valid_during_success={sum(1 for ok, _, _ in during_results if ok)}/"
            f"{len(during_results)} ratio={during_ratio:.3f} "
            f"(min={args.min_during_success_ratio:.3f})"
        )
        print(
            f"valid_after_success={sum(1 for ok, _, _ in after_results if ok)}/"
            f"{len(after_results)} ratio={after_ratio:.3f} "
            f"(min={args.min_after_success_ratio:.3f})"
        )
        print(
            f"during_p95_ms={during_p95_ms:.1f} "
            f"(max={args.max_during_p95_ms:.1f})"
        )

        failures: list[str] = []
        if during_ratio < args.min_during_success_ratio:
            failures.append(
                f"during success ratio too low: {during_ratio:.3f} < {args.min_during_success_ratio:.3f}"
            )
        if after_ratio < args.min_after_success_ratio:
            failures.append(
                f"after success ratio too low: {after_ratio:.3f} < {args.min_after_success_ratio:.3f}"
            )
        if during_p95_ms > args.max_during_p95_ms:
            failures.append(
                f"during p95 latency too high: {during_p95_ms:.1f}ms > {args.max_during_p95_ms:.1f}ms"
            )

        if failures:
            print("\nFAIL")
            for f in failures:
                print(f"- {f}")

            sample_errors = [status for ok, _, status in during_results if not ok][:5]
            if sample_errors:
                print("sample_during_errors:")
                for err in sample_errors:
                    print(f"- {err}")
            return 1

        print("\nPASS")
        return 0
    finally:
        if proxy_proc is not None:
            terminate_process(proxy_proc, "proxy")
        if upstream_proc is not None:
            terminate_process(upstream_proc, "upstream")
        if config.exists():
            config.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
