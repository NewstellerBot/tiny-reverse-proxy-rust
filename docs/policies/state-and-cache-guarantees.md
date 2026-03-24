# State and Cache Guarantees

## Authoritative state

These are control-plane truth and must be recoverable from store-backed state:

- projects
- virtual keys
- managed providers
- project policy/routing/tool/prompt/rollout configuration
- config revision state

## Advisory state

These are node-local optimizations and must never become the sole source of correctness:

- prompt-cache routing memory
- semantic cache entries
- in-memory virtual-key and provider overlays
- runtime readiness/drain state

## Required guarantees

- Cache loss must not corrupt correctness.
- Reloads must replace stale advisory state, not merge it indefinitely.
- Multi-node correctness must depend on store-backed state, not node-local memory.
- Anything that can diverge across nodes must be treated as advisory only.

## Operator expectations

- Caches may improve latency and routing quality, but they are not control-plane truth.
- Rebuilding node-local state from store must be safe after restart, deploy, or failover.
