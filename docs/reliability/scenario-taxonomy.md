# Scenario Taxonomy

Every reliability row in this program must use one of these families. Do not
invent ad hoc categories.

## Families

### API compatibility

- `/chat/completions`
- `/responses`
- `/embeddings`
- `/images`
- `/audio`
- `/batches`
- `/files`

### Streaming correctness

- SSE chunk ordering
- content deltas
- tool call streaming
- final usage/completion events
- early disconnect behavior

### Retry and fallback classification

- timeout
- 429
- upstream 5xx
- auth failure
- malformed request or malformed upstream payload
- non-retryable provider error

### Load, soak, and degradation

- sustained throughput
- latency growth over time
- memory growth over time
- brownout activation
- admission-control rejection behavior

### Multi-node and shared-state correctness

- rolling restart
- node-local cache loss
- shared store continuity
- config revision propagation
- readiness/drain behavior

### Tool and MCP reliability

- disconnect/reconnect
- stale session recovery
- tool timeout or budget exhaustion
- retryable tool failures
- auth or session expiry

### Config and control-plane safety

- validate-before-apply
- apply/rollback
- import/export
- revision history
- recovery after startup

### Observability and operator contract

- metrics
- logs
- OTEL
- routing/failover visibility
- overload/brownout visibility

### Deployment lifecycle

- startup validation
- liveness/readiness
- graceful drain
- release-gated smoke
- fault-injection validation

## Required row fields

Each row in the parity matrix must include:

- `scenario_id`
- `family`
- `description`
- `reference_projects`
- `why_it_matters`
- `current_status`
- `current_repo_evidence`
- `target_validation_mode`
- `priority`

## Allowed values

### `current_status`

- `covered`
- `partial`
- `missing`
- `not_applicable`

### `target_validation_mode`

- `unit`
- `integration`
- `live_smoke`
- `benchmark`
- `soak`
- `manual_release_check`

### `priority`

- `P0`
- `P1`
- `P2`

## Interpretation rules

- `covered` means the repo already has an automated artifact that directly
  exercises the scenario.
- `partial` means the repo covers part of the scenario but not the full
  operator contract or failure matrix.
- `missing` means the repo does not currently have a meaningful automated
  artifact for the scenario.
- `not_applicable` is reserved for future use; prefer `missing` unless the
  scenario genuinely does not make sense for this project.

- `unit` is only for logic that can be validated without a full proxy/gateway
  instance.
- `integration` is the default for deterministic black-box correctness checks.
- `live_smoke` is for secret-backed real-provider validation.
- `benchmark` is for latency, throughput, CPU, or RSS evidence.
- `soak` is for long-duration behavior and degradation detection.
- `manual_release_check` is for high-cost or human-reviewed release checks that
  do not belong on every branch.
