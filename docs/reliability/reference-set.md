# Reference Set

This program compares Tiny Proxy against notable LLM gateways, but each project
is used for a specific kind of evidence.

## Locked comparison set

| Project | How we use it | Reliability signals we borrow | Primary sources |
|---|---|---|---|
| LiteLLM | OSS test and issue reference | Long-run soak expectations, proxy compatibility bugs, endpoint normalization edge cases, release-channel distinction between `main` and `stable` | [README](https://github.com/BerriAI/litellm), [docs](https://docs.litellm.ai/), [performance degradation issue #6345](https://github.com/BerriAI/litellm/issues/6345), [buffered TTS streaming issue #14891](https://github.com/BerriAI/litellm/issues/14891), [Responses incompatibility issue #12231](https://github.com/BerriAI/litellm/issues/12231), [batch/files issue #5396](https://github.com/BerriAI/litellm/issues/5396) |
| Bifrost | OSS runtime and governance reference | MCP reconnect/session/auth behavior, governance/operator controls, multi-node and transport reliability expectations | [MCP overview](https://docs.getbifrost.ai/mcp/overview), [connecting to servers](https://docs.getbifrost.ai/mcp/connecting-to-servers), [agent mode](https://docs.getbifrost.ai/mcp/agent-mode), [budget and limits](https://docs.getbifrost.ai/features/governance/budget-and-limits), [adaptive load balancing](https://docs.getbifrost.ai/enterprise/intelligent-load-balancing) |
| Cloudflare AI Gateway | Black-box operator-contract reference | Retries, fallbacks, rate limits, caching, dynamic routing, metadata, OTEL, deployment limits | [overview](https://developers.cloudflare.com/ai-gateway/), [features](https://developers.cloudflare.com/ai-gateway/features/), [dynamic routing](https://developers.cloudflare.com/ai-gateway/features/dynamic-routing/), [OTEL integration](https://developers.cloudflare.com/ai-gateway/observability/otel-integration/), [limits](https://developers.cloudflare.com/ai-gateway/reference/limits/) |
| OpenRouter | Black-box routing and provider-selection reference | Request-level provider order, fallback controls, ZDR/privacy constraints, provider sticky routing, cache-affinity expectations | [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection), [prompt caching](https://openrouter.ai/docs/features/prompt-caching), [zero data retention](https://openrouter.ai/docs/features/zdr), [latency and performance](https://openrouter.ai/docs/features/latency-and-performance) |
| Portkey | Black-box routing and observability reference | Automatic retries, fallback status visibility, gateway configs, tracing across retries/fallbacks | [AI Gateway overview](https://portkey.ai/docs/product/ai-gateway), [automatic retries](https://portkey.ai/docs/product/ai-gateway/automatic-retries), [fallbacks](https://portkey.ai/docs/product/ai-gateway/fallbacks), [configs](https://portkey.ai/docs/product/ai-gateway/configs), [logs and analytics](https://portkey.ai/docs/portkey-features/observability/logs-and-analytics) |
| Helicone | Workflow and observability reference | Retry semantics, prompt experimentation, operator-facing request analysis | [retries](https://docs.helicone.ai/features/advanced-usage/retries), [prompts](https://www.helicone.ai/prompts), [experiments](https://docs.helicone.ai/features/experiments) |
| Langfuse | Workflow and control-plane reference | Prompt management, datasets, eval loops, control-plane change-management expectations | [prompts](https://langfuse.com/docs/prompts/get-started), [evaluation](https://langfuse.com/docs/evaluation/overview), [datasets](https://langfuse.com/docs/datasets/overview) |

## How to use these references

- LiteLLM and Bifrost are the only first-class OSS code/test references.
- Cloudflare AI Gateway, OpenRouter, Portkey, Helicone, and Langfuse are used
  to sharpen the scenario matrix and operator contract.
- If a hosted product documents an operator behavior, we can add a matrix row
  for that behavior without treating the vendor's exact UI or API shape as a
  requirement.

## What is explicitly out of scope

Do not create parity rows for:

- hosted-only dashboards or workflow UIs
- billing, seat management, or enterprise admin surfaces
- vendor-specific hosted integrations that do not change gateway correctness
- marketing-only claims that do not map to a concrete runtime behavior
