#!/usr/bin/env python3
"""Shared helpers for stress/soak scripts."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_proxy_binary(explicit_bin: str | None = None) -> Path:
    if explicit_bin:
        path = Path(explicit_bin).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"proxy binary not found: {path}")
        return path

    path = repo_root() / "target" / "debug" / "tiny-reverse-proxy"
    if path.exists():
        return path

    print("[setup] building tiny-reverse-proxy debug binary...")
    subprocess.run(
        ["cargo", "build", "-p", "tiny-reverse-proxy"],
        cwd=repo_root(),
        check=True,
    )
    if not path.exists():
        raise RuntimeError(f"build succeeded but binary missing at {path}")
    return path


def ensure_h3_spike_client_binary() -> Path:
    path = repo_root() / "target" / "debug" / "examples" / "h3_spike_client"
    print("[setup] building proxy-core h3_spike_client example...")
    subprocess.run(
        ["cargo", "build", "-p", "proxy-core", "--example", "h3_spike_client"],
        cwd=repo_root(),
        check=True,
    )
    if not path.exists():
        raise RuntimeError(f"build succeeded but h3_spike_client missing at {path}")
    return path


def start_upstream(port: int) -> subprocess.Popen:
    upstream_script = repo_root() / "bench" / "upstream_threaded.py"
    return subprocess.Popen(
        [sys.executable, str(upstream_script), str(port)],
        cwd=repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def start_proxy(binary: Path, config_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("RUST_LOG", "error")
    return subprocess.Popen(
        [str(binary), str(config_path)],
        cwd=repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def terminate_process(proc: subprocess.Popen, name: str, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def wait_for_tcp_port(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for TCP {host}:{port}")


def wait_for_http_status(
    url: str,
    expected_status: str,
    timeout_sec: float,
    insecure_tls: bool = False,
    http3_only: bool = False,
) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        cmd = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}"]
        if insecure_tls:
            cmd.append("-k")
        if http3_only:
            cmd.append("--http3-only")
        cmd.append(url)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == expected_status:
            return
        time.sleep(0.1)
    raise TimeoutError(
        f"timed out waiting for {url} to return {expected_status}"
    )


def write_temp_config(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    try:
        tmp.write(content)
    finally:
        tmp.close()
    return Path(tmp.name)


def has_curl_http3() -> bool:
    result = subprocess.run(["curl", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return "HTTP3" in result.stdout or "http3" in result.stdout


def process_rss_kb(pid: int) -> int | None:
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    if not output:
        return None
    try:
        return int(output)
    except ValueError:
        return None
