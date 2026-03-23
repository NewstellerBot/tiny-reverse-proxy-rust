# Stress / Soak Tests

These scripts are intentionally **assertive**: they fail with non-zero exit codes when behavior deviates from expectations.

## 1) HTTP/3 Large-Body Spike

File: `http3_large_body_spike.py`

Purpose:
- Sends oversized HTTP/3 POST bodies at spike concurrency.
- Expects `413 Payload Too Large` for almost all spike requests.
- Verifies HTTP/3 GET health still returns `200` after the spike.
- Checks proxy RSS growth against a budget.

Run:

```bash
python3 tests/stress/http3_large_body_spike.py
```

Notes:
- Uses `curl --http3-only` when available.
- If local curl lacks HTTP/3 support, it automatically falls back to a native Rust HTTP/3 client (`proxy-core/examples/h3_spike_client.rs`).
- The fallback client uses `--insecure` for local test runs (equivalent to curl `-k`).
- Uses TLS auto mode in proxy config.

## 2) PROXY Slow-Header Soak

File: `proxy_protocol_slow_header_soak.py`

Purpose:
- Opens many slow/incomplete PROXY protocol headers.
- Sends valid PROXY+HTTP requests during and after load.
- Expects high success ratio and bounded p95 latency.

Run:

```bash
python3 tests/stress/proxy_protocol_slow_header_soak.py
```

## Tuning

Both scripts expose CLI flags (`--help`) to tune:
- total load (`--requests`, `--attack-connections`)
- concurrency (`--concurrency`, `--attack-concurrency`)
- SLO thresholds (success ratio / p95 / RSS growth)
