# Benchmarking

## When benchmark evidence is required

Hot-path changes must include benchmark evidence or a written note explaining why the change should
not measurably affect the hot path.

Hot-path files include at minimum:

- `crates/proxy-core/src/handlers/proxy.rs`
- `crates/proxy-core/src/runtime.rs`
- `crates/plugin-llm-gateway/src/virtual_keys.rs`
- `crates/plugin-llm-gateway/src/tool_runtime.rs`
- cache modules

## Required scenarios

Benchmark evidence should cover the affected subset of:

- plain proxy request path
- gateway virtual-key request path
- `/v1/responses` non-streaming
- `/v1/responses` strict streaming
- cache hit and miss paths
- provider failover path
- translated image/audio/embedding path when affected

## Expectations

- Benchmark claims should be before/after comparisons, not standalone numbers.
- If a change knowingly regresses performance, the PR should say why the tradeoff is acceptable.
- If a benchmark is not yet automated, include the exact command or workflow used to gather it.
