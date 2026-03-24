# Tiny Proxy

![banner](web/ui-kit/img/banner.png)

A high-performance reverse proxy built in Rust with HTTP/3 support, an LLM gateway plugin, and a built-in web dashboard.

## Architecture

The project is a Cargo workspace with three crates:

| Crate | Type | Purpose |
|---|---|---|
| `proxy-core` | lib | Core proxy library — config, routing, handlers, plugin API, rate limiting, caching, compression, TLS, HTTP/3, metrics, circuit breaker, health checks |
| `plugin-llm-gateway` | lib | LLM gateway plugin — cost tracking, provider failover, rate limiting, virtual keys, streaming support |
| `tiny-reverse-proxy` | bin | Binary entrypoint |

## Features

- HTTP/1.1, HTTP/2, and HTTP/3 (QUIC) support
- TLS termination
- Glob-based route matching
- Plugin system with pre/post request hooks
- Rate limiting and circuit breaker
- Response caching and compression
- PROXY protocol support
- Built-in web dashboard with live metrics
- LLM gateway plugin with per-key cost tracking, budgets, and provider failover
- Local process probes on `/_trp/livez` and `/_trp/readyz`
- Built-in admission control and brownout thresholds for pre-deployment hardening

## Docs

- [Management API](/Users/krystian/code/tiny-reverse-proxy-rust/docs/management-api.md)
- [Deployment Guide](/Users/krystian/code/tiny-reverse-proxy-rust/docs/deployment.md)
- [Reliability Program](/Users/krystian/code/tiny-reverse-proxy-rust/docs/reliability/README.md)
- [Semantic Safety V0](/Users/krystian/code/tiny-reverse-proxy-rust/docs/semantic-safety-v0.md)
- [Project Policies](/Users/krystian/code/tiny-reverse-proxy-rust/docs/policies/README.md)

## Getting Started

### Prerequisites

Install [Rust](https://www.rust-lang.org/tools/install), or use the Nix dev shell:

```sh
nix develop  # or direnv allow
```

### Build & Run

```sh
cargo build --release
./target/release/tiny-reverse-proxy --config config.toml
```

To enable the LLM gateway plugin:

```sh
cargo build --release --features plugin-llm-gateway
```

### Tests

```sh
cargo test --workspace
```

### Live OpenAI Realtime Smoke Tests

These tests are ignored by default because they make live network calls to the
OpenAI Realtime WebSocket API and require credentials.

```sh
OPENAI_API_KEY=... \
cargo test -p tiny-reverse-proxy openai_realtime_proxy_connects_and_receives_session_created -- --ignored --nocapture

OPENAI_API_KEY=... \
cargo test -p tiny-reverse-proxy openai_realtime_proxy_can_generate_text_response -- --ignored --nocapture
```

Optional environment variables:

- `OPENAI_REALTIME_MODEL` to override the default model (`gpt-realtime`)
- `OPENAI_REALTIME_TIMEOUT_SECS` to increase the per-event wait timeout
- `OPENAI_ORGANIZATION` / `OPENAI_ORG_ID`
- `OPENAI_PROJECT` / `OPENAI_PROJECT_ID`

### Live OpenAI Responses Smoke Tests

These tests are ignored by default because they make live network calls to the
OpenAI Responses API through the local proxy. They will read `OPENAI_API_KEY`
from the environment or the repo-root `.env`.

```sh
cargo test -p tiny-reverse-proxy openai_responses_proxy_can_generate_text_response -- --ignored --nocapture

cargo test -p tiny-reverse-proxy openai_responses_proxy_can_stream_text_response -- --ignored --nocapture
```

Optional environment variables:

- `OPENAI_RESPONSES_MODEL` to override the default model (`gpt-4.1-mini`)
- `OPENAI_RESPONSES_TIMEOUT_SECS` to increase the request/body timeout
- `OPENAI_ORGANIZATION` / `OPENAI_ORG_ID`
- `OPENAI_PROJECT` / `OPENAI_PROJECT_ID`

### Live Gateway Responses Smoke Tests

These tests are ignored by default and require the `plugin-llm-gateway`
feature. They mint a real virtual key, send `/v1/responses` through the gateway
plugin chain, and verify the live OpenAI response comes back through that
managed path.

```sh
cargo test -p tiny-reverse-proxy --features plugin-llm-gateway gateway_virtual_key_responses_proxy_can_generate_text_response -- --ignored --nocapture

cargo test -p tiny-reverse-proxy --features plugin-llm-gateway gateway_virtual_key_responses_proxy_can_stream_text_response -- --ignored --nocapture
```

### Releases

Versions should be cut through the GitHub Actions `Release` workflow, not by
creating tags manually. Releases are cut from `release/<major>.<minor>`
branches, using prerelease versions like `1.4.0-rc.1` for RCs and a final
version from the same branch after soak. The workflow runs the live gateway
`/v1/responses` smoke and a deterministic release-validation suite before it
creates the version tag and GitHub release. Configure the
`release-live-openai` environment with the OpenAI secrets and required
reviewers so the live smoke stays protected and explicit.

### Python Realtime Smoke Test

For a standalone smoke test outside Rust's test harness:

```sh
export OPENAI_API_KEY=...
uv run --with websockets python3 scripts/openai_realtime_smoke.py --start-proxy --scenario smoke
```

For a heavier proxy-driven workload test:

```sh
export OPENAI_API_KEY=...
uv run --with websockets python3 scripts/openai_realtime_smoke.py --start-proxy --scenario workload
```

To test OpenAI directly instead of a local proxy:

```sh
export OPENAI_API_KEY=...
uv run --with websockets python3 scripts/openai_realtime_smoke.py --url wss://api.openai.com/v1/realtime --scenario smoke
```

### Stress Testing

```sh
python3 tests/stress/http3_large_body_spike.py
python3 tests/stress/proxy_protocol_slow_header_soak.py
```
