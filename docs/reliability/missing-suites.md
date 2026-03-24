# Missing Suites

This backlog converts the parity matrix into concrete new validation suites.

The goal is to make the next implementation wave decision-complete. File
locations, harness choice, and CI placement are chosen here so the implementer
does not need to invent structure.

## Wave 1

These suites should land first.

### 1. Soak / release-candidate suite

Purpose:

- detect latency drift, memory drift, and degradation under long-running traffic
- validate the gateway under a release-candidate shape, not just short
  benchmarks

Implementation shape:

- add `bench/reliability/README.md`
- add `bench/reliability/soak_runner.py`
- add `bench/reliability/fixtures/` for mock upstream profiles
- use `uv run python ...` for all runner entrypoints

Scenarios:

- pass-through
- `/responses` non-streaming
- `/responses` strict streaming
- provider failover under periodic injected 429/5xx

Captured outputs:

- p50/p95/p99 latency snapshots over time windows
- error-rate trend over time windows
- sampled RSS and CPU trend
- brownout/admission-control activation counts

Placement:

- maintainer-triggered on `release/*`
- initial duration: 2 hours
- later extension target: 12 hours for stable candidates

### 2. Retry-classification integration suite

Purpose:

- turn retry/fallback behavior into an explicit matrix instead of accidental
  coverage through unrelated tests

Implementation shape:

- add `crates/plugin-llm-gateway/tests/retry_classification_integration.rs`
- use local fake upstreams to emit each error class deterministically
- assert both user-visible result and internal counters/headers/log state

Required matrix:

- timeout
- 429
- 500/502/503/504 class errors
- auth failure
- malformed upstream JSON/body
- non-retryable provider semantic error

Assertions:

- retry happened or did not happen
- fallback happened or did not happen
- final status code/body is correct
- relevant metrics/log fields moved as expected

Placement:

- release-branch CI after the first implementation
- may also run in normal CI if execution time stays modest

### 3. Multi-node correctness suite

Purpose:

- prove that shared-store operation works the way the docs claim

Implementation shape:

- add `tests/reliability/docker-compose.multinode.yml`
- add `crates/tiny-reverse-proxy/tests/multi_node_reliability.rs`
- run 3 proxy nodes against shared Postgres
- keep node-local caches enabled

Required scenarios:

- rolling restart while traffic continues
- readiness drain removes a node before shutdown
- config revision apply/rollback with all nodes converging
- virtual-key/store reload replacement behavior across nodes
- node-local cache loss without correctness drift

Assertions:

- no partial control-plane truth across nodes
- readiness reflects rollout/drain state correctly
- cache loss changes latency at most, not correctness

Placement:

- subset on release-branch CI
- full suite on maintainer-triggered validation

### 4. Compatibility goldens suite

Purpose:

- capture the highest-signal compatibility bugs seen in other gateways and lock
  them down in fixtures

Implementation shape:

- add `crates/plugin-llm-gateway/tests/compatibility_goldens.rs`
- add `tests/reliability/goldens/` for request and response fixtures
- fixtures should be scenario-driven, not vendor-labeled

Required golden groups:

- `/responses`
- `/batches`
- translated image/audio/embedding paths
- tool streaming

Fixture sources:

- official provider docs
- LiteLLM public issue patterns
- current repo behavior when already known-good

Placement:

- release-branch CI
- candidate to move into ordinary CI once stable and fast

### 5. Observability contract suite

Purpose:

- verify that the operator-facing reliability surfaces are not just present but
  meaningful

Implementation shape:

- add `crates/plugin-llm-gateway/tests/observability_contract.rs`
- add targeted fake upstream/tool failure profiles
- assert metrics, log fields, status surfaces, and OTEL-compatible signals

Required scenarios:

- provider failover
- overload/brownout activation
- admission-control reject
- retry exhaustion
- control-plane degradation that does not block hot-path serving

Assertions:

- counters and labels increment correctly
- management/status surfaces expose the degraded state
- request logs preserve enough detail to explain the incident

Placement:

- release-branch CI
- selected fast cases may later move to normal CI

## Wave 2

These suites come next, after Wave 1 is stable and useful.

### Tool-and-MCP recovery suite

- add `crates/plugin-llm-gateway/tests/mcp_recovery_integration.rs`
- cover disconnect/reconnect, expired auth/session recovery, budget exhaustion,
  and retryable transport errors

### Startup-and-recovery suite

- add `crates/plugin-llm-gateway/tests/startup_recovery_integration.rs`
- cover cold boot with valid store, malformed store state, and partial
  control-plane unavailability with safe data-plane behavior

### Longer stable-candidate soak

- extend `bench/reliability/soak_runner.py`
- move from 2-hour RC soak to 12-hour stable-candidate soak once:
  - data collection format is stable
  - thresholds are evidence-based
  - shorter soak runs are catching real regressions

## Dependency order

Implement in this order:

1. retry-classification integration suite
2. observability contract suite
3. compatibility goldens suite
4. multi-node correctness suite
5. 2-hour RC soak suite
6. Wave 2 suites

That order keeps the early wins deterministic and branch-friendly before moving
into longer-running infrastructure-heavy validation.
