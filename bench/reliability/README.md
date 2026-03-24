# Release-candidate soak harness

This harness runs a mixed mock-plus-live soak against a local `tiny-reverse-proxy`
instance that it starts itself.

What it does:
- mock lane at sustained low/medium concurrency using local flaky + healthy upstreams
- live lane at low continuity-only request rates against the OpenAI-compatible `/v1/responses`
  path when `OPENAI_API_KEY` is available
- collects latency, error-rate, RSS/CPU samples, retry/failover counters, and brownout/admission
  status snapshots

Defaults:
- mock chat/response traffic runs continuously
- live non-streaming requests run every 60 seconds
- live streaming requests run every 120 seconds
- first runs are evidence-gathering and should be read as trend reports, not strict latency gates

Run locally with:

```bash
uv run python bench/reliability/soak_runner.py --duration-secs 300
```

If `OPENAI_API_KEY` is set, the harness automatically enables the live lane. Otherwise it runs
mock-only and reports that the live lane was skipped.
