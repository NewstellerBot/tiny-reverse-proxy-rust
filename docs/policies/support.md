# Support Matrix

## Platform support

- Tier 1: Linux x86_64 on stable Rust, validated in GitHub Actions on Ubuntu
- Tier 2: none currently
- Community/best-effort: macOS, Windows, and other Linux targets

## Store support

- Tier 1: SQLite
- Tier 2: PostgreSQL and MySQL

SQLite is the default and most exercised path. PostgreSQL and MySQL are intentionally supported,
but they remain a step below SQLite until they receive broader CI and release validation.

## Provider support

- Stable: OpenAI family
- Preview/best-effort: Anthropic, Gemini, OpenRouter
- Experimental: Custom provider family

Stable means the provider family is part of the default support story. Preview means the family is
intentionally shipped but may still have endpoint-specific caveats. Experimental means the project
accepts the configuration shape but does not claim broad compatibility.

## Endpoint posture

- Stable:
  - core proxy path
  - virtual keys
  - provider failover
  - `/v1/chat/completions`
  - `/v1/responses` in strict mode
  - readiness/liveness
  - admission control
- Preview:
  - composed `/v1/responses` streaming
  - control-plane import
  - provider-surface rewrites that depend on non-native translation
- Best-effort/advanced:
  - realtime
  - advanced MCP orchestration
  - semantic safety

## Deployment posture

- Supported for development: single-node local/dev
- Recommended for real use: multiple proxy instances behind a load balancer with a shared store
- Unsupported:
  - public management API exposure without protection
  - multi-node setups that rely on node-local control-plane truth

See [Deployment Topology](deployment-topology.md) for the deployment model this support matrix
assumes.
