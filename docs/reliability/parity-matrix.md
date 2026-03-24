# Parity Matrix

This matrix tracks reliability behaviors we actually care about, not feature
count parity.

## API compatibility

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| api-chat-completions | API compatibility | OpenAI-style `/v1/chat/completions` request/response compatibility through routing and virtual keys | LiteLLM, Bifrost, Cloudflare AI Gateway | This is the baseline path many clients still use. | covered | `crates/proxy-core/tests/conformance.rs`; `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs` | integration | P0 |
| api-responses | API compatibility | `/v1/responses` non-streaming correctness through the gateway path | LiteLLM, Cloudflare AI Gateway | This is the main modern OpenAI surface and already a release gate. | covered | `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs`; `crates/tiny-reverse-proxy/tests/live_gateway_responses.rs`; `.github/workflows/release.yml` | integration | P0 |
| api-embeddings | API compatibility | embeddings routing and provider translation correctness | LiteLLM, OpenRouter | Embeddings are frequently normalized incorrectly across providers. | partial | `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs` | integration | P1 |
| api-images | API compatibility | image generation/edit/variation correctness across native and translated providers | LiteLLM, OpenRouter | Multipart and translated image paths are high-risk for silent incompatibility. | partial | `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs` | integration | P1 |
| api-audio | API compatibility | audio speech/transcription/translation routing and translation correctness | LiteLLM, OpenRouter | Audio paths often regress around content types and streamed bodies. | partial | `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs` | integration | P1 |
| api-batches | API compatibility | batch create/retrieve/cancel semantics and target-surface validation | LiteLLM, Cloudflare AI Gateway | Batch correctness depends on provider affinity and endpoint compatibility. | partial | `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs` | integration | P1 |
| api-files | API compatibility | `/v1/files` routing and provider-surface correctness | LiteLLM, Cloudflare AI Gateway | File surfaces are control-plane-adjacent and easy to under-test. | partial | `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs` | integration | P1 |

## Streaming correctness

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| stream-sse-ordering | Streaming correctness | SSE event ordering stays valid under provider and tool streaming | LiteLLM, Bifrost, Cloudflare AI Gateway | Broken event order creates client-visible corruption. | partial | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs`; `crates/tiny-reverse-proxy/tests/live_gateway_responses.rs` | integration | P0 |
| stream-content-deltas | Streaming correctness | content deltas are forwarded without buffering or shape drift | LiteLLM, Cloudflare AI Gateway | Buffering or coalescing deltas changes UX and latency. | partial | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs`; `bench/oss-gateway-shootout/README.md` | integration | P0 |
| stream-tool-calls | Streaming correctness | tool-call streaming remains in-band and protocol-correct | Bifrost, OpenRouter | Agent workloads depend on this staying stable. | covered | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs` | integration | P0 |
| stream-final-usage | Streaming correctness | terminal usage and completion events remain present and well-formed | LiteLLM, Cloudflare AI Gateway | Billing, analytics, and downstream clients depend on final events. | partial | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs`; `crates/tiny-reverse-proxy/tests/live_gateway_responses.rs` | integration | P1 |
| stream-early-disconnect | Streaming correctness | client disconnect during stream does not corrupt gateway state or wedge resources | LiteLLM, Cloudflare AI Gateway | This is a common production failure mode under real clients. | missing | no dedicated deterministic test today | integration | P0 |

## Retry and fallback classification

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| retry-timeout | Retry and fallback classification | timeout responses retry/fallback exactly when intended | Cloudflare AI Gateway, Portkey, LiteLLM | Retry behavior must be explicit, not accidental. | partial | `crates/proxy-core/tests/conformance.rs`; `.github/workflows/release.yml` | integration | P0 |
| retry-429 | Retry and fallback classification | 429s trigger the right retry/fallback path and produce clear terminal status | Portkey, OpenRouter, LiteLLM | Rate limiting is a normal operating mode, not an edge case. | partial | `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs`; `crates/proxy-core/tests/rate_limit_integration.rs` | integration | P0 |
| retry-5xx | Retry and fallback classification | upstream 5xx behavior is classified correctly for retry/fallback | Cloudflare AI Gateway, Portkey, LiteLLM | Wrong 5xx classification amplifies incidents. | partial | `crates/plugin-llm-gateway/tests/llm_gateway_integration.rs`; `crates/plugin-llm-gateway/tests/management_integration.rs` | integration | P0 |
| retry-auth-failure | Retry and fallback classification | auth failures never trigger unsafe retries and surface clearly | Portkey, OpenRouter, LiteLLM | Retrying auth failures wastes budget and obscures root cause. | partial | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs`; `crates/plugin-llm-gateway/tests/management_integration.rs` | integration | P1 |
| retry-malformed-upstream | Retry and fallback classification | malformed upstream payloads fail closed and are classified distinctly | LiteLLM, Cloudflare AI Gateway | Gateways must not silently reinterpret bad provider payloads. | partial | `crates/proxy-core/tests/conformance.rs` | integration | P1 |
| retry-non-retryable-provider-error | Retry and fallback classification | semantic or protocol-level non-retryable errors stop immediately | Portkey, OpenRouter, LiteLLM | This keeps retries from masking user or provider contract errors. | partial | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs` | integration | P0 |

## Load, soak, and degradation

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| load-sustained-throughput | Load, soak, and degradation | sustained throughput and latency at bounded CPU/memory limits | LiteLLM, Bifrost, Cloudflare AI Gateway | Fast steady-state performance is part of the product thesis. | partial | `bench/oss-gateway-shootout/README.md`; `bench/oss-gateway-shootout/run.py` | benchmark | P0 |
| soak-latency-growth | Load, soak, and degradation | latency drift over long runs stays bounded | LiteLLM | Long-running degradation is a common proxy failure mode. | missing | no long-duration soak artifact today | soak | P0 |
| soak-memory-growth | Load, soak, and degradation | RSS growth and leak-like behavior are tracked over time | LiteLLM | Memory drift is often invisible in short benchmarks. | missing | no long-duration memory-growth gate today | soak | P0 |
| degrade-brownout | Load, soak, and degradation | brownout mode activates predictably before hard rejection | Cloudflare AI Gateway, Portkey | Optional features should shed before the gateway collapses. | partial | `docs/deployment.md`; `crates/proxy-core/src/runtime.rs`; release validation docs | integration | P0 |
| degrade-admission-control | Load, soak, and degradation | admission control rejects excess load early and deterministically | Cloudflare AI Gateway, Portkey | This is a core anti-cascade control. | covered | `crates/proxy-core/src/handlers/proxy.rs`; `.github/workflows/release.yml` | integration | P0 |

## Multi-node and shared-state correctness

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| multinode-rolling-restart | Multi-node and shared-state correctness | rolling restart does not interrupt safe traffic serving | Bifrost, Cloudflare AI Gateway | This is the minimum serious deployment behavior. | missing | no multi-node test harness today | integration | P0 |
| multinode-cache-loss | Multi-node and shared-state correctness | losing node-local caches does not cause correctness drift | OpenRouter, Cloudflare AI Gateway | Node-local cache state must stay advisory only. | missing | state/cache guarantees are documented, but not exercised across nodes | integration | P0 |
| multinode-shared-store | Multi-node and shared-state correctness | shared Postgres/MySQL state survives multiple proxy nodes and restarts | Bifrost, Cloudflare AI Gateway | Shared control-plane state is a core production assumption. | partial | `crates/plugin-llm-gateway/tests/real_db_store.rs`; `crates/plugin-llm-gateway/tests/store_and_api.rs`; `.github/workflows/ci.yml` | integration | P0 |
| multinode-config-propagation | Multi-node and shared-state correctness | config revisions propagate safely without partial truth | Portkey, Langfuse | Multi-node control-plane correctness is change-safety critical. | partial | `crates/plugin-llm-gateway/tests/store_and_api.rs` | integration | P0 |
| multinode-readiness-drain | Multi-node and shared-state correctness | readiness flips and drain behavior support LB-safe rollout | Cloudflare AI Gateway, Bifrost | Load balancers rely on readiness semantics, not hope. | partial | `crates/tiny-reverse-proxy/tests/shutdown.rs`; `.github/workflows/release.yml`; `docs/deployment.md` | integration | P0 |

## Tool and MCP reliability

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| mcp-disconnect-reconnect | Tool and MCP reliability | transient MCP disconnects recover cleanly without unsafe stale state | Bifrost | MCP transport failures are expected in production. | partial | `crates/plugin-llm-gateway/tests/management_integration.rs`; `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs` | integration | P1 |
| mcp-stale-session-recovery | Tool and MCP reliability | stale or expired MCP sessions recover deterministically | Bifrost | Session recovery is core to long-lived tool runtimes. | covered | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs`; `crates/plugin-llm-gateway/tests/management_integration.rs` | integration | P0 |
| mcp-tool-budget-exhaustion | Tool and MCP reliability | tool call/output/time budgets fail closed and remain visible | Bifrost, Cloudflare AI Gateway | Tool safety depends on bounded execution. | covered | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs`; `crates/plugin-llm-gateway/tests/management_integration.rs` | integration | P0 |
| mcp-retryable-tool-failure | Tool and MCP reliability | retryable tool failures retry only where intended | Bifrost, Portkey | Tool retries can amplify incidents if misclassified. | covered | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs` | integration | P1 |
| mcp-auth-expiry | Tool and MCP reliability | auth or session expiry recovers without operator guesswork | Bifrost | Protected MCP servers make auth expiry normal, not rare. | covered | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs`; `crates/plugin-llm-gateway/tests/management_integration.rs` | integration | P1 |

## Config and control-plane safety

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| config-validate-before-apply | Config and control-plane safety | invalid config is rejected before activation | Portkey, Langfuse | Change safety starts with validation. | covered | `crates/plugin-llm-gateway/tests/store_and_api.rs` | integration | P0 |
| config-apply-rollback | Config and control-plane safety | runtime-affecting config applies atomically and rolls back cleanly | Portkey, Langfuse, Cloudflare AI Gateway | Partial config state is a reliability failure. | covered | `crates/plugin-llm-gateway/tests/store_and_api.rs` | integration | P0 |
| config-import-export | Config and control-plane safety | control-plane export/import works for backup and restore flows | Langfuse, Portkey | Disaster recovery depends on this path being real. | partial | `crates/plugin-llm-gateway/tests/store_and_api.rs`; preview policy docs | integration | P1 |
| config-revision-history | Config and control-plane safety | revision history keeps active and last-known-good state explicit | Portkey, Langfuse | Operators need deterministic rollback targets. | partial | `crates/plugin-llm-gateway/tests/store_and_api.rs`; management APIs | integration | P1 |
| config-startup-recovery | Config and control-plane safety | startup recovery preserves safe runtime state after restart | Langfuse, Cloudflare AI Gateway | Restart behavior matters as much as steady-state behavior. | partial | `crates/plugin-llm-gateway/tests/store_and_api.rs`; `.github/workflows/release.yml` | integration | P0 |

## Observability and operator contract

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| obs-metrics | Observability and operator contract | metrics expose failover, usage, and runtime state correctly | Cloudflare AI Gateway, Portkey, Helicone | Operators need counters they can alert on. | covered | `crates/proxy-core/tests/metrics_integration.rs`; `crates/plugin-llm-gateway/tests/metrics_integration.rs` | integration | P1 |
| obs-logs | Observability and operator contract | logs and request history preserve enough detail to debug retries/failover | Helicone, Portkey, Langfuse | Reliability work fails if incidents are not explainable. | partial | `crates/plugin-llm-gateway/tests/management_integration.rs`; request-log APIs | integration | P1 |
| obs-otel | Observability and operator contract | OTEL surfaces compile and emit the expected gateway/provider signals | Cloudflare AI Gateway, Langfuse | OTEL is part of the promised operator surface. | partial | `crates/plugin-llm-gateway/tests/otel_compilation.rs`; `.github/workflows/ci.yml` | integration | P1 |
| obs-routing-visibility | Observability and operator contract | provider health and routing debug surfaces explain failover choices | Cloudflare AI Gateway, OpenRouter, Portkey | Operators need to know why a provider was chosen. | covered | `crates/plugin-llm-gateway/tests/management_integration.rs`; `docs/management-api.md` | integration | P0 |
| obs-overload-visibility | Observability and operator contract | overload, admission-control, and brownout states are externally visible | Cloudflare AI Gateway, Portkey | Shedding load without visibility is operationally weak. | partial | `docs/deployment.md`; `crates/proxy-core/src/runtime.rs`; current probes and metrics | integration | P1 |

## Deployment lifecycle

| scenario_id | family | description | reference_projects | why_it_matters | current_status | current_repo_evidence | target_validation_mode | priority |
|---|---|---|---|---|---|---|---|---|
| deploy-startup-validation | Deployment lifecycle | startup fails clearly on unsafe runtime or plugin configuration | Bifrost, Cloudflare AI Gateway | Invalid startup state should fail early and loudly. | covered | `crates/plugin-llm-gateway/tests/tool_runtime_integration.rs`; `.github/workflows/release.yml` | integration | P0 |
| deploy-liveness-readiness | Deployment lifecycle | liveness and readiness express process safety, not upstream health guesses | Cloudflare AI Gateway | Correct probe semantics are foundational to deployment safety. | covered | `crates/proxy-core/src/runtime.rs`; `.github/workflows/release.yml`; `docs/deployment.md` | integration | P0 |
| deploy-graceful-drain | Deployment lifecycle | in-flight requests complete while new traffic drains away | Cloudflare AI Gateway, Bifrost | Rolling deploys depend on this. | covered | `crates/tiny-reverse-proxy/tests/shutdown.rs` | integration | P0 |
| deploy-release-gated-smoke | Deployment lifecycle | release creation is blocked on deterministic checks plus live gateway smoke | LiteLLM, Cloudflare AI Gateway | Version cuts should be guarded by real evidence. | covered | `.github/workflows/release.yml`; `.github/workflows/live-openai-responses-smoke.yml` | live_smoke | P0 |
| deploy-fault-injection | Deployment lifecycle | deterministic fault-injection scenarios run before release promotion | Cloudflare AI Gateway, Portkey, LiteLLM | This is the missing bridge between unit confidence and production confidence. | partial | `.github/workflows/release.yml` has deterministic checks, but not a dedicated fault-injection suite | integration | P0 |
