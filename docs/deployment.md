# Deployment Guide

This document covers the practical deployment split the gateway now supports:

- SQLite for local development and single-node testing
- Postgres or MySQL for shared or production control-plane state

## Local development

For local work, use SQLite and keep the full gateway in one process.

Minimal shape:

```toml
port = 8080
management_api_port = 9090
management_api_token = "$TRP_BOOTSTRAP_ADMIN_TOKEN"
store_url = "sqlite:///tmp/tiny-proxy-dev.db"

[paths]
"/**" = ["https://api.openai.com"]

[[providers]]
name = "openai"
api_key = "$OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"
models = ["gpt-4o", "gpt-4o-mini"]
family = "openai"

[providers.surfaces]
tools = "openai"
responses = "openai_compatible"

[[plugins]]
name = "prompt_cache"
enabled = true

[plugins.config]
anthropic_default_scope = "auto"

[[plugins]]
name = "tool_runtime"
enabled = true

[plugins.config]
tool_timeout_ms = 5000
max_round_trips = 8

[reliability]
max_inflight_requests = 2048
brownout_inflight_requests = 1536
retry_budget_per_request = 2
```

Notes:

- `virtual_keys` is created automatically when `providers` are configured.
- Managed tools and prompt cache are intended to run behind virtual keys, not
  direct provider keys.
- If `content_filter` and `semantic_safety` are both enabled, `content_filter`
  must come first. Startup now fails on unsafe ordering.
- `/_trp/livez` and `/_trp/readyz` are reserved local process probes. They are
  separate from upstream `[health_check]`, which still controls origin health
  selection.
- `reliability.max_inflight_requests` is a hard local admission cap.
  `reliability.brownout_inflight_requests` activates local brownout mode first,
  which disables optional hot-path features like prompt cache, semantic cache,
  semantic safety export, and managed tools before hard rejection starts.

## Production state backends

Use Postgres or MySQL once multiple operators or longer-lived governance state
matter.

Supported store URLs:

- `postgres://...`
- `mysql://...`
- `sqlite:///...`

Current gateway coverage includes:

- virtual keys
- request logs
- model pricing / usage
- project policies
- routing rules
- safety policies
- semantic-safety policies
- project tools

The gateway now expects the current schema as the baseline on SQLite, Postgres,
and MySQL connection paths.

## Management API

Expose `management_api_port` only to trusted operators or an internal network.

Recommended baseline:

- require bearer auth
- keep bootstrap admin tokens out of checked-in config
- use the management API for virtual key creation instead of hardcoding runtime
  secrets into app config

See [management-api.md](/Users/krystian/code/tiny-reverse-proxy-rust/docs/management-api.md)
for the current endpoint surface.

## Claude Code On A LAN

If you want other machines on your local network to use Claude Code through
this gateway without sharing your real Anthropic key, use a provider-backed
runtime key:

1. Start from [configs/claude-code-lan.example.toml](/Users/krystian/code/tiny-reverse-proxy-rust/configs/claude-code-lan.example.toml).
2. Set `ANTHROPIC_API_KEY` on the proxy host only.
3. Set a strong `TRP_BOOTSTRAP_ADMIN_TOKEN` on the proxy host before startup.
4. Run the proxy with `--features plugin-llm-gateway`.

Example:

```sh
export ANTHROPIC_API_KEY=...
export TRP_BOOTSTRAP_ADMIN_TOKEN='replace-with-a-long-random-secret'

cargo run --release --features plugin-llm-gateway -- \
  --config configs/claude-code-lan.example.toml
```

The main proxy listener already binds to `0.0.0.0`, so other devices on your
LAN can reach `http://<proxy-host>:8888`. The management API stays on
`127.0.0.1:9090`, which means project and runtime-key administration stays on
the proxy host.

Create a project and a runtime key from the proxy host:

```sh
curl -s http://127.0.0.1:9090/api/v1/projects \
  -H "Authorization: Bearer $TRP_BOOTSTRAP_ADMIN_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"project_id":"claude-code-lan","name":"Claude Code LAN"}'

curl -s http://127.0.0.1:9090/api/v1/keys \
  -H "Authorization: Bearer $TRP_BOOTSTRAP_ADMIN_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"project_id":"claude-code-lan","name":"lan-client","provider_name":"anthropic"}'
```

The second call returns a managed runtime key in the `key` field. Put that key
on each Claude Code client instead of your real Anthropic credential.

Anthropic's Claude Code gateway docs currently use:

- `ANTHROPIC_BASE_URL` for the unified gateway endpoint
- `ANTHROPIC_AUTH_TOKEN` for the token Claude Code sends as `Authorization`

On each client machine:

```sh
export ANTHROPIC_BASE_URL="http://<proxy-host>:8888"
export ANTHROPIC_AUTH_TOKEN="<returned sk-trp-... key>"
```

Claude Code expects the gateway to expose Anthropic-compatible `/v1/...`
endpoints and preserve Anthropic request headers such as
`anthropic-version`/`anthropic-beta`. This config does that while the
`virtual_keys` plugin swaps the managed runtime key for your real upstream
`x-api-key` before forwarding to Anthropic.

Relevant Claude Code docs:

- [Use an LLM gateway](https://docs.anthropic.com/en/docs/claude-code/llm-gateway)
- [Bedrock, Vertex, and proxy configuration](https://docs.anthropic.com/en/docs/claude-code/bedrock-vertex-proxies)

Recommended LAN posture:

- keep `allow_direct_provider_keys = false`
- keep `management_api_port` off the LAN
- restrict the proxy port to your local subnet with a firewall or router ACL
- use TLS if your LAN is not fully trusted

## Observability

Recommended production posture:

- enable Prometheus scraping for both core and gateway metrics
- build and run with OpenTelemetry when you need provider-turn, tool-call, and
  semantic-evaluation spans
- keep routing debug request headers disabled by default and only enable them
  for live debugging
- scrape the local process probes on the main listener:
  - `GET /_trp/livez`
  - `GET /_trp/readyz`
- keep readiness tied to local process safety only:
  - startup validation complete
  - management/metrics listeners successfully bound
  - not currently draining for shutdown

Relevant visibility surfaces:

- `/metrics`
- `/_trp/livez`
- `/_trp/readyz`
- `/api/v1/providers`
- `/api/v1/providers/health`
- `/api/v1/tool-runtime/status`
- `/api/v1/prompt-cache/status`

## Multi-node baseline

Minimum serious production topology:

- multiple proxy instances behind a load balancer
- Postgres or MySQL as shared control-plane state
- `management_api_port` exposed only on an operator network
- node-local caches treated as advisory only
- readiness/liveness integrated with rollout and drain behavior

Operational rules:

- do not rely on in-memory cache state for correctness or control-plane truth
- use `/_trp/readyz` for load balancer readiness, not upstream `[health_check]`
- treat `SIGHUP` as route-only reload; provider/plugin/runtime changes still
  require a restart
- keep release cuts on the Actions `Release` workflow so the deterministic test
  suite and live gateway smoke both gate version creation
- isolate the management API from public ingress

Policy references:

- [Support Matrix](/Users/krystian/code/tiny-reverse-proxy-rust/docs/policies/support.md)
- [Deployment Topology](/Users/krystian/code/tiny-reverse-proxy-rust/docs/policies/deployment-topology.md)
- [State and Cache Guarantees](/Users/krystian/code/tiny-reverse-proxy-rust/docs/policies/state-and-cache-guarantees.md)
- [Release Policy](/Users/krystian/code/tiny-reverse-proxy-rust/docs/policies/release.md)
- [Reliability Program](/Users/krystian/code/tiny-reverse-proxy-rust/docs/reliability/README.md)

## Semantic safety deployment

Semantic safety v0 stays observe-only and runs as a separate service. Keep it
on a dedicated GPU host and treat gateway ordering rules as mandatory so local
redaction happens before any remote semantic export.

See [semantic-safety-v0.md](/Users/krystian/code/tiny-reverse-proxy-rust/docs/semantic-safety-v0.md)
for the dedicated service setup.

## Validation before rollout

Minimum operator checks:

- run `cargo test -p plugin-llm-gateway --tests`
- run the real DB gateway tests against disposable Postgres/MySQL instances when
  changing persistence or migrations
- run a local proxy smoke against a real provider key before changing provider
  routing or managed runtime behavior

The repo CI now covers:

- gateway unit/integration tests
- OpenTelemetry compilation coverage
- real Postgres/MySQL round-trip gateway store tests with service containers
- release-gated deterministic validation for probes, retries, shutdown, and
  startup checks
- release-gated live gateway `/v1/responses` smoke behind a protected
  environment

Use [docs/reliability/release-gates.md](/Users/krystian/code/tiny-reverse-proxy-rust/docs/reliability/release-gates.md)
for the lane map covering PR CI, release-branch validation, release hard gates,
and maintainer-triggered soak runs.
