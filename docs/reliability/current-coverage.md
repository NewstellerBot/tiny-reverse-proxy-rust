# Current Coverage

This document maps the repo as it exists today onto the reliability matrix.

## Current posture

The current repo is:

- strong on black-box gateway integration, management API behavior, provider
  failover, tool runtime behavior, prompt/semantic cache routing, and
  release-gated live `/v1/responses`
- partial on long-duration soak, multi-node shared-store correctness, explicit
  retry-classification matrices, and observability contract validation
- weak or missing on release-branch fault injection, long-run degradation
  detection, and competitor-derived compatibility goldens

## Existing evidence by artifact

| Artifact | What it already covers | Matrix families it feeds |
|---|---|---|
| `crates/proxy-core/tests/conformance.rs` | upstream timeout handling, malformed upstream behavior, 502 behavior, load distribution | API compatibility, retry classification |
| `crates/proxy-core/tests/health.rs` | upstream health selection and fallback to healthy upstreams | retry/fallback, deployment lifecycle |
| `crates/proxy-core/tests/rate_limit_integration.rs` | hard 429 behavior after budget exhaustion | retry classification, overload behavior |
| `crates/proxy-core/tests/cache_integration.rs` | core cache hit/miss correctness | API compatibility, load/degradation |
| `crates/proxy-core/tests/metrics_integration.rs` | core Prometheus surfaces | observability |
| `crates/tiny-reverse-proxy/tests/shutdown.rs` | graceful shutdown, completion of in-flight work, new-connection refusal | deployment lifecycle, readiness/drain |
| `crates/tiny-reverse-proxy/tests/live_gateway_responses.rs` | live non-streaming and streaming `/v1/responses` through the gateway path | API compatibility, streaming correctness, release smoke |
| `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs` | endpoint routing, surface translation, failover on 429/5xx, prompt/semantic cache routing | API compatibility, retry/fallback, cache-affinity behavior |
| `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs` | tool/runtime correctness, streaming tool events, MCP retry/session recovery, tool budgets, startup validation | streaming correctness, tool/MCP reliability, deployment lifecycle |
| `crates/plugin-llm-gateway/tests/management_integration.rs` | provider health visibility, failover debug surfaces, MCP operator controls, runtime status views | observability, tool/MCP reliability, control-plane safety |
| `crates/plugin-llm-gateway/tests/store_and_api.rs` | config revision apply/validate/rollback, import/export, startup recovery, control-plane APIs | control-plane safety |
| `crates/plugin-llm-gateway/tests/real_db_store.rs` | real Postgres/MySQL round trips for store-backed gateway state | shared-store continuity |
| `crates/plugin-llm-gateway/tests/otel_compilation.rs` | OTEL build and compile coverage | observability |
| `bench/oss-gateway-shootout/run.py` and `bench/oss-gateway-shootout/README.md` | throughput, latency, RSS, CPU across `tiny-proxy`, LiteLLM, Bifrost, and direct upstream | benchmark posture, competitor baseline |
| `.github/workflows/ci.yml` | PR/push validation for fmt, clippy, workspace tests, OTEL compilation, real DB store tests | baseline CI placement |
| `.github/workflows/release.yml` | deterministic release validation, probe/admission tests, retry tests, shutdown checks, startup checks, live smoke dependency | release-gate posture |
| `.github/workflows/live-openai-responses-smoke.yml` | protected live gateway `/v1/responses` smoke | live smoke |

## Coverage by scenario family

### Strong today

- API compatibility for core request paths and modern `/v1/responses`
- streaming tool-loop correctness
- provider failover and routing visibility
- MCP session recovery and tool-budget enforcement
- config validate/apply/rollback round trips
- release-gated live gateway `/v1/responses`

### Partial today

- non-chat surface normalization breadth
- retry classification across all error classes
- observability contract beyond “metric exists”
- shared-store correctness across more than one node
- overload behavior beyond single-node admission rejection

### Missing or underdeveloped today

- long-duration soak and degradation detection
- multi-node rolling restart and node-local cache-loss correctness
- compatibility goldens derived from public ecosystem issues
- dedicated release-branch fault-injection suites
- explicit observability contract verification for overload and retry exhaustion

## Practical interpretation

The repo already has enough deterministic black-box coverage to support
day-to-day development. The next reliability gap is not "write more unit tests."
It is:

- longer-running evidence
- multi-node evidence
- clearer retry/fallback classification
- stronger operator-contract verification
- compatibility cases chosen from real gateway failure reports
