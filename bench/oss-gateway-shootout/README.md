# OSS Gateway Shootout

Reproducible benchmark harness for comparing `tiny-reverse-proxy` against other
open source LLM gateways under the same Docker resource limits.

## Current targets

- `tiny-reverse-proxy`
- `litellm`
- `bifrost`
- `direct` upstream baseline

## What it measures

- throughput (`Requests/sec`)
- latency (`avg`, `p50`, `p95`, `p99`)
- sampled CPU usage
- sampled RSS / memory usage
- image reference and resolved image ID
- exact Docker CPU/memory limits used for the run

## Methodology

- Every gateway runs in its own Docker container.
- Every gateway gets the same `--cpus` and `--memory` limits.
- Every gateway proxies the same mock OpenAI-compatible upstream.
- Gateway containers talk to the mock through its host-published endpoint
  (`host.docker.internal`) instead of a Docker bridge alias. That avoids a
  container-to-container networking artifact that can dominate short local runs
  and drown out actual proxy overhead.
- Every run uses one shared request shape across all targets for the selected
  scenario.
- Gateway containers default to Docker `--log-driver=none`, the mock upstream
  runs in `--quiet` mode, and gateway-specific per-request logging is reduced
  so the benchmark focuses on proxying work instead of stdout overhead.
- Results include the generated gateway config files so the comparison is
  auditable.
- Generated `tiny-proxy` configs use canonical `family` + `surfaces` provider
  definitions, not the old flat compatibility fields.

The harness now supports seven scenarios:

- `pass-through`: standard non-streaming chat completion
- `streaming`: OpenAI-style `stream=true` chat completion
- `prompt-cache`: provider-native cache controls, with `tiny-proxy` booting the
  real gateway `prompt_cache` runtime before the benchmark starts
- `prompt-cache-affinity`: a mixed-provider cache-routing benchmark where
  `tiny-proxy` warms one provider, flips fallback order to prefer another, and
  then measures whether durable prompt-cache affinity keeps routing on the
  warmed provider
- `prompt-cache-affinity-routing-only`: the same warm-and-reroute flow as
  `prompt-cache-affinity`, but with prompt-cache routing-hint persistence
  disabled so the benchmark isolates routing overhead from durable store writes
- `semantic-cache-affinity`: a mixed-provider semantic-cache benchmark where
  `tiny-proxy` warms one provider, flips fallback order, and then measures the
  cache-locality routing plus gateway-served semantic-cache hit path. This
  compares the Rust in-process cache-hit path with a Python mock direct baseline,
  so it is not a pure proxy-overhead apples-to-apples result.
- `tool-round-trip`: a managed tool loop benchmark for `tiny-proxy` plus a
  direct self-orchestrated baseline

Support matrix:

- `pass-through`: `direct`, `tiny-proxy`, `litellm`, `bifrost`
- `streaming`: `direct`, `tiny-proxy`, `litellm`, `bifrost`
- `prompt-cache`: `direct`, `tiny-proxy`, `litellm`, `bifrost`
- `prompt-cache-affinity`: `direct`, `tiny-proxy`
- `prompt-cache-affinity-routing-only`: `direct`, `tiny-proxy`
- `semantic-cache-affinity`: `direct`, `tiny-proxy`
- `tool-round-trip`: `direct`, `tiny-proxy`

For `prompt-cache`, `prompt-cache-affinity`,
`prompt-cache-affinity-routing-only`, `semantic-cache-affinity`, and
`tool-round-trip`, `tiny-proxy` runs with providers, management API, SQLite
state, and a runtime key/bootstrap flow so the benchmark actually exercises the
shipped gateway features rather than forwarding inert JSON fields upstream.

## Prerequisites

- Docker daemon running
- `hey` installed
- `uv` installed

## Run

```sh
uv run python bench/oss-gateway-shootout/run.py
```

Useful flags:

```sh
uv run python bench/oss-gateway-shootout/run.py \
  --targets direct tiny-proxy \
  --scenario tool-round-trip \
  --cpus 1.0 \
  --memory 512m \
  --duration 15s \
  --concurrency 32 \
  --gateway-log-driver none
```

To only render configs / commands without touching Docker:

```sh
uv run python bench/oss-gateway-shootout/run.py --dry-run
```

## Output

Artifacts land under:

```text
bench/oss-gateway-shootout/results/<timestamp>/
```

Each run writes:

- `results.json` raw machine-readable output
- `SUMMARY.md` human-readable summary
- `configs/` generated config files used by each gateway
- benchmark scenario metadata so pass-through and streaming runs are directly comparable

## Notes

- `tiny-reverse-proxy` is built into a local Docker image from
  [Dockerfile.tiny-proxy](/Users/krystian/code/tiny-reverse-proxy-rust/bench/oss-gateway-shootout/Dockerfile.tiny-proxy).
- LiteLLM uses the official image documented at
  [docs.litellm.ai](https://docs.litellm.ai/).
- Bifrost uses the official image and `config.json`/`APP_DIR` flow documented at
  [docs.getbifrost.ai](https://docs.getbifrost.ai/quickstart/gateway/setting-up).
- The runner auto-detects the working request path for LiteLLM and Bifrost
  because their OpenAI-compatible endpoints differ slightly by project/version.
