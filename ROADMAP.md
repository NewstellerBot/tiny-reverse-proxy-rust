# Tiny Proxy Roadmap

Updated: 2026-03-12

## What Is In The Repo Today

### Core proxy

- Rust reverse proxy with HTTP/1.1, HTTP/2, and HTTP/3 support
- TLS termination, route matching, compression, caching, circuit breaker, health checks, and PROXY protocol support
- Built-in dashboard and Prometheus metrics
- Plugin system for request/response interception and policy enforcement

### LLM gateway

- Provider configuration and model aliasing
- Virtual keys with project scoping and optional direct-provider-key blocking
- Cost tracking, per-model pricing, budgets, budget alerts, and request logging
- RPM and TPM aware rate limiting
- Provider failover and routing controls
- Provider health visibility, routing debug headers, and cooldown reason reporting
- Provider-aware prompt-cache controls and cache-usage visibility
- Semantic cache controls, status, and request-time reuse for supported request shapes
- Effective runtime policy inspection across project, provider, and virtual-key scopes
- Streaming-aware usage extraction and OpenAI Realtime smoke coverage
- Expanded provider capability routing for Responses API, reasoning, image input, audio input, audio output, realtime, and prompt-cache-specific controls

### Governance and management API

- Management API for status, usage, model pricing, providers, keys, logs, projects, principals, role bindings, and tokens
- Project policies for budgets, limits, adaptive routing, and fallback order
- Project routing rules with path/model/header/streaming match conditions
- Governance history, policy diffs, and RBAC access introspection
- Session-scoped summaries over request logs and tool activity
- Persistent storage for SQLite, Postgres, and MySQL

### Safety and policy

- Content filter plugin with block, observe, and redact modes
- Regex and detector-based secret/sensitive-content matching
- Semantic safety v0 as a separate Rust service with project policy sync and status APIs
- Request-log audit fields for content-filter and semantic-safety outcomes
- Startup enforcement for unsafe plugin ordering

### Tools and agent runtime

- Project-scoped tool registry
- Multi-provider tool protocol support from the start via explicit provider config
- Non-streaming and streaming managed tool loops for OpenAI-style and Anthropic-style providers
- Built-in `web_search` and `arxiv_search` executors
- Remote MCP-backed managed tools with startup discovery and session recovery
- MCP server and per-tool call and time budgets with runtime status visibility for budget exhaustion
- Generic `webhook` executor with tool-result safety replay and audit aggregation
- Project/key tool approval policy with allow-all, deny-all, and allow-list modes
- Tool runtime status and provider capability visibility in the management API, including live MCP reachability, retry/session settings, and per-server call stats

### Prompts, datasets, and evals

- Project-scoped prompt registry with environments, rollout metadata, active version switching, request-time prompt resolution, persisted staged rollout workflows, request-time canary execution for applied prompt rollouts, policy-driven canary stage advance/rollback flows, and autonomous live-canary evaluation with auto-advance and auto-rollback triggers
- Project-scoped datasets and dataset items for replay/regression inputs
- Persisted eval runs with per-item results, latency, usage, cost, and summary metrics
- Async/background eval runs with startup recovery, richer output matchers including structured JSON-path checks, external, OpenAI-compatible, and Anthropic-compatible judge evaluators, and run-to-run comparison views with context diffs, saved rollout policies, prompt-promotion workflows, rollout presets, and gate verdicts

### Benchmarks and validation

- OSS gateway benchmark harness against `tiny-reverse-proxy`, LiteLLM, Bifrost, and direct upstream
- Pass-through and streaming benchmark scenarios in the OSS shootout harness
- Competitor-derived reliability matrix and missing-suite backlog under
  [docs/reliability/README.md](./docs/reliability/README.md)
- Real DB round-trip coverage for gateway store paths
- CI coverage for OpenTelemetry compilation and real Postgres/MySQL gateway store tests
- Realtime smoke tests and tool-runtime black-box integration tests

## Product Thesis

This project should become the fastest self-hosted AI gateway and control plane that teams can trust in production.

The differentiator should not be "another OpenAI-compatible proxy." The differentiator should be:

- lower overhead and better streaming correctness than Python-heavy gateways
- stronger governance, safety, and routing than a simple pass-through proxy
- an agent/tool runtime that stays provider-agnostic
- a self-hosted control-plane story that is simpler than the large hosted platforms

## Market Signals Driving The Roadmap

These signals are based on competitor docs, community pain points collected in [PAIN_POINTS.md](./PAIN_POINTS.md), and our own benchmark direction in [bench/oss-gateway-shootout/README.md](./bench/oss-gateway-shootout/README.md).

### 1. Baseline gateway features are already table stakes

Cloudflare AI Gateway, LiteLLM, and OpenRouter already cover the baseline set users now expect: analytics/logging, rate limits, retries/fallbacks, caching, routing, and spend controls. That means we should treat multi-provider routing, budgets, guardrails, and caching as required infrastructure, not as the end-state product.

### 2. The market is moving up-stack into control-plane workflows

Bifrost is pushing MCP and agent-mode workflows. Langfuse and Helicone are pushing prompt management, experiments, and eval workflows around the gateway/runtime layer. The implication is clear: teams no longer want only a proxy. They want a control plane for prompts, tools, routing, safety, and evaluation.

This is an inference from competitor direction, not a direct statement from any one vendor.

### 3. Reliability and low overhead are still a real opening

Community reports continue to surface around proxy buffering, broken streaming behavior, and long-running performance degradation in popular gateway stacks. That matters because "fast and boring" is still a strong product position, especially for self-hosted deployments that do not want a sprawling Python control plane in the request path.

### 4. Tooling and agent execution are becoming first-class, not optional

Once teams start using tools, they need policy, auditability, and provider abstraction around tool execution. Raw provider tool calling is not enough. The gateway becomes the place where tool allowlists, timeouts, budgets, and safety policies need to live.

## Roadmap

Most of the original "finish the gateway" work is now done. The roadmap should no longer treat streaming tools, built-in search executors, routing visibility, prompt registry basics, datasets, eval storage, semantic cache basics, governance history, or session summaries as future work. Those are now part of the shipped baseline.

The remaining roadmap is below, ordered by what still creates the most product leverage.

## Active Priorities

Target: next 1-2 months

The original Evals v2 gap is now closed: judge adapters cover webhook, OpenAI-compatible, and Anthropic-compatible endpoints, and live prompt canaries can advance or roll back themselves from fresh eval evidence. The remaining roadmap is below.

As of 2026-03-13, the biggest remaining parity gaps versus Cloudflare AI
Gateway, LiteLLM, OpenRouter, and Bifrost are no longer prompt registry, eval,
dataset, or basic tool/runtime work. Those are already in the shipped
baseline. The gaps are now more specific:

- OpenRouter-style request-level provider selection and privacy/cost controls
- LiteLLM-style endpoint translation and normalization breadth
- Bifrost-style MCP transport, auth, and exposure breadth
- Cloudflare-style request metadata, provider control-plane, and per-request
  operator controls

### 1. MCP Operational Controls

- MCP runtime status now exposes discovery refreshes, last discovery outcome, budget exhaustion details, session reinitialization timestamps, manual session reset state, and operator-facing MCP health summaries
- Operators can now force MCP rediscovery with `POST /api/v1/tool-runtime/mcp/{server}/refresh`, clear a stale cached session with `DELETE /api/v1/tool-runtime/mcp/{server}/session`, and disable or re-enable an MCP server live with `POST /api/v1/tool-runtime/mcp/{server}/disable` / `POST /api/v1/tool-runtime/mcp/{server}/enable`
- Remaining work is narrower now:
  - keep improving MCP execution-path logging and failure explanations for operators
  - expand budgets beyond call counts into richer time/cost policies where it actually changes operator safety

Why this is next:
The managed tool runtime is already credible, and the basic operator control surface is now in place. What is still missing is the deeper production layer around richer MCP policy and richer time/cost controls.

### 2. Cache-Aware Routing And Benchmarks

- Move warmed prompt-cache state from a post-routing reorder into explicit governance scoring
  - Treat prompt-cache warmth as a bounded provider-affinity bonus instead of a simple move-to-front override
  - Keep provider-specific prompt-cache protocols explicit while making the routing effect explainable
- Expose cache-driven routing reasons to operators
  - Add prompt-cache affinity details to provider-health and routing-debug output
  - Make it clear when a provider was chosen because cache was warm versus because of fallback order or health
- Persist prompt-cache routing memory across restarts
  - Keep the hot lookup path in memory
  - Add durable backing for warmed prompt-cache provider hints so restart does not reset provider affinity
- Add a negative and freshness-aware cache signal
  - Decay stale warmed entries over time
  - Differentiate stronger write/read warmth from weaker stale or recently missed cache state
- Extend cache-aware routing beyond prompt cache
  - Reuse the same routing-affinity model for durable semantic-cache locality
  - Avoid keeping prompt-cache and semantic-cache routing as separate one-off behaviors
- Semantic cache is now store-backed when a gateway store is configured, with restart-safe reloads and SQLite/Postgres/MySQL backend support
- Add matched benchmark scenarios for cache-aware routing
  - Cold request versus warm request
  - Warm request after gateway restart
  - Cache-affinity preference versus project fallback order
  - Mixed-provider prompt-cache and semantic-cache paths

Why this is next:
Cache behavior is now implemented, and the benchmark harness already covers prompt-cache and managed tool round trips in addition to pass-through and plain streaming. The remaining work is to turn cache locality into a first-class routing signal, make that behavior visible to operators, and prove the effect with matched benchmarks.

### 3. Longer-Lived Agent Sessions

- Durable session rollups and session `status` / `state` / `metadata` are now persisted in the store and exposed through the management API
- Session lifecycle now has explicit management controls:
  - `POST /api/v1/sessions/{id}/transition`
  - `POST /api/v1/sessions/{id}/heartbeat`
  - persisted transition timestamps, reasons, heartbeat times, and lease expiry
- Session ownership and cancellation now have an explicit control surface:
  - `POST /api/v1/sessions/{id}/claim`
  - `POST /api/v1/sessions/{id}/release`
  - `POST /api/v1/sessions/{id}/cancel`
  - persisted owner, lease, and cancel-request state for long-running runtimes
- Session ownership handoff now has an explicit transfer contract:
  - `POST /api/v1/sessions/{id}/handoff`
  - `POST /api/v1/sessions/{id}/accept`
  - persisted pending handoff target, request time, and reason
- Session watchability and lease-aware takeover now have an explicit runtime surface:
  - `GET /api/v1/sessions` with ownership/recovery filters for operator views
  - `POST /api/v1/sessions/{id}/takeover` for stale-lease resume or forced override
- Session orchestration now has a durable change feed:
  - `GET /api/v1/sessions/{id}/events` for ordered session mutation and request activity
  - `GET /api/v1/sessions/{id}/wait` for simple long-poll wakeups on the next session change
- Stale-session reconciliation now runs in the gateway runtime and can also be
  forced with `POST /api/v1/sessions/{id}/reconcile`
  - expired owner leases are reconciled into `paused` recovery-required sessions
  - pending cancels are finalized once no active owner remains
- Remaining work is now the harder orchestration layer:
  - improve realtime/session orchestration beyond long-poll/event-feed coordination
  - move from lease/cancel/handoff coordination into richer long-lived agent execution semantics

Why this is next:
The gateway now has durable session continuity plus explicit lifecycle, ownership, and cancellation control instead of only request-log-derived summaries. The next harder, more defensible layer is long-lived agent correctness and realtime orchestration.

## Near-Term Follow-Ons

Target: next quarter

### 1. Request-Level Provider Controls

- Close the biggest OpenRouter-style gap in the routing layer
- Add a request-level provider policy object for runtime overrides such as:
  - provider order
  - allow or deny fallbacks
  - require-parameters behavior
  - provider allow/ignore filters
  - quantization or tier preferences where providers expose them
  - max-price and latency/throughput thresholds where that meaningfully changes routing
  - privacy and collection flags such as zero-data-retention or no-training style constraints
  - provider-specific beta/header toggles where users need them without hard-forking config
- Keep project-level routing rules and adaptive routing as the default, but let
  request-level intent narrow the eligible provider set safely
- Add black-box tests for request-level provider policy interactions with failover,
  prompt cache, semantic cache, and capability routing

### 2. Endpoint Translation And Normalization

- Close the biggest LiteLLM-style gap in provider abstraction
- Expand from capability routing into true request and response normalization for
  non-chat surfaces
- Prioritize:
  - `/v1/images` and image-generation parity
  - `/v1/batches` parity
  - deeper audio surface normalization beyond routing-only checks
  - embeddings request and response normalization where providers diverge materially
- Keep errors and streaming semantics consistent across providers where possible,
  not only request acceptance
- Treat black-box surface tests as mandatory for every newly normalized endpoint

### 3. MCP Surface Expansion

- Close the biggest Bifrost-style gap in agent-runtime connectivity
- Expand MCP support beyond the current outbound HTTP/session-controlled runtime
- Prioritize:
  - stdio MCP client support
  - SSE-based MCP client support where servers expose it
  - OAuth 2.0, PKCE, and token-refresh flows for protected MCP servers
  - dynamic client registration where MCP deployments expect it
  - optional inbound MCP gateway exposure so external clients can use the gateway as an MCP surface
  - richer tool hosting and filtering contracts where they materially improve operability
- Keep operator controls and health visibility first-class as the transport/auth
  matrix expands

### 4. Request Metadata And Provider Control Plane

- Close the biggest Cloudflare-style operator-surface gap
- Add request-scoped metadata tags that persist into logs, sessions, and analytics
- Add per-request or per-model custom cost overrides where operators need to
  reflect nonstandard pricing without rewriting global pricing tables
- Add provider CRUD and provider-policy management in the control plane instead of
  relying only on static config plus visibility endpoints
- Keep OTel and store-backed analytics as the base, but add clearer export and
  operator query surfaces where teams need external reporting

### 5. Stronger Governance And Rollout Controls

- Keep pushing effective-policy visibility across project/provider/key/runtime scopes
- Add more rollout-oriented policy surfaces where operators need deterministic explanations
- Tighten audit history around prompt, route, and tool changes

### 6. Low-Overhead Competitive Parity Validation

- Add more matched-feature benchmark scenarios against competitor gateways
- Keep validating that new features do not erode the low-overhead position
- Treat benchmark regressions as product regressions, not only engineering regressions

## Phase 4: Strategic Adjacent Bet

Target: after the gateway/control-plane story is solid

### AI traffic defense on ingress

- Add crawler and AI-bot policy enforcement at the proxy layer
- Add stronger fingerprinting and behavior-based controls
- Add purpose-aware policy where possible, instead of only blunt block/allow behavior

Why this is interesting:
The core proxy already owns TLS, request inspection, routing, and observability. That makes AI ingress control a plausible adjacent product. It should stay a secondary track until the LLM gateway roadmap above is solid.

## What We Should Not Do Yet

- Do not build a broad SaaS control plane before the self-hosted product is clearly better than the current OSS alternatives.
- Do not add a large number of built-in tool executors before `webhook`, `web_search`, and `arxiv_search` are production-safe.
- Do not make semantic safety blocking-by-default before observe-only quality is validated.
- Do not lead with a UI-heavy prompt IDE before prompt registry, eval, and audit primitives exist.

## Success Criteria

- The benchmark harness continues to show low proxy overhead relative to other OSS gateways.
- Every provider/tool path has a black-box end-to-end test.
- Operators can answer: who used what model, via which provider, with which tool, at what cost, under which policy.
- A team can run the full gateway with SQLite locally and Postgres/MySQL in production without migration surprises.
- The project becomes known for correctness under streaming, failover, and tool execution, not only for feature count.

## Reference Points

Local project references:

- [README.md](./README.md)
- [docs/semantic-safety-v0.md](./docs/semantic-safety-v0.md)
- [PAIN_POINTS.md](./PAIN_POINTS.md)
- [bench/oss-gateway-shootout/README.md](./bench/oss-gateway-shootout/README.md)

External references used to shape this roadmap:

- Cloudflare AI Gateway overview: <https://developers.cloudflare.com/ai-gateway/>
- Cloudflare custom metadata: <https://developers.cloudflare.com/ai-gateway/configuration/custom-metadata/>
- Cloudflare custom costs: <https://developers.cloudflare.com/ai-gateway/configuration/custom-costs/>
- Cloudflare custom providers: <https://developers.cloudflare.com/ai-gateway/configuration/custom-providers/>
- LiteLLM virtual keys: <https://docs.litellm.ai/docs/proxy/virtual_keys>
- LiteLLM caching: <https://docs.litellm.ai/docs/proxy/caching>
- LiteLLM guardrails quick start: <https://docs.litellm.ai/docs/proxy/guardrails/quick_start>
- LiteLLM load balancing: <https://docs.litellm.ai/docs/proxy/load_balancing>
- LiteLLM docs: <https://docs.litellm.ai/>
- OpenRouter provider routing: <https://openrouter.ai/docs/guides/routing/provider-selection>
- OpenRouter prompt caching: <https://openrouter.ai/docs/features/prompt-caching>
- Bifrost MCP overview: <https://docs.getbifrost.ai/mcp/overview>
- Bifrost MCP server connections: <https://docs.getbifrost.ai/mcp/connecting-to-servers>
- Bifrost tool hosting: <https://docs.getbifrost.ai/mcp/tool-hosting>
- Bifrost agent mode: <https://docs.getbifrost.ai/mcp/agent-mode>
- Langfuse prompt management: <https://langfuse.com/docs/prompts/get-started>
- Langfuse evaluation: <https://langfuse.com/docs/evaluation/overview>
- Langfuse datasets: <https://langfuse.com/docs/datasets/overview>
- Helicone prompts: <https://www.helicone.ai/prompts>
- Helicone experiments: <https://docs.helicone.ai/features/experiments>
- LiteLLM GitHub issue on performance degradation over time: <https://github.com/BerriAI/litellm/issues/6345>
- LiteLLM GitHub issue on buffered TTS streaming through the proxy: <https://github.com/BerriAI/litellm/issues/14891>
