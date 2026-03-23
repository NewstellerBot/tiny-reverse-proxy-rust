#!/usr/bin/env python3
"""
Fetch current model pricing from OpenRouter and generate a TOML
model_costs table for the cost_tracker plugin.

Usage:
  # Print TOML snippet to stdout (paste into your config):
  python3 scripts/sync_model_prices.py

  # Filter to specific providers:
  python3 scripts/sync_model_prices.py --providers openai anthropic google

  # Write a complete example config:
  python3 scripts/sync_model_prices.py --full > examples/llm_gateway.toml

  # Show all available providers:
  python3 scripts/sync_model_prices.py --list-providers
"""

import argparse
import json
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError

OPENROUTER_API = "https://openrouter.ai/api/v1/models"

# Map OpenRouter provider prefixes to short names.
PROVIDER_ORDER = [
    "openai",
    "anthropic",
    "google",
    "mistralai",
    "meta-llama",
    "deepseek",
    "cohere",
    "qwen",
]


def fetch_models():
    """Fetch the model list from OpenRouter."""
    req = Request(OPENROUTER_API, headers={"User-Agent": "tiny-reverse-proxy/sync-prices"})
    try:
        resp = urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        return data.get("data", [])
    except (URLError, json.JSONDecodeError) as e:
        print(f"Error fetching from OpenRouter: {e}", file=sys.stderr)
        sys.exit(1)


def parse_model(entry):
    """Extract (provider, model_id, short_name, input_per_1k, output_per_1k) from an entry."""
    model_id = entry.get("id", "")
    pricing = entry.get("pricing", {})

    prompt_price = pricing.get("prompt")
    completion_price = pricing.get("completion")
    if prompt_price is None or completion_price is None:
        return None

    try:
        # OpenRouter prices are per-token strings, convert to per-1k floats
        input_per_1k = float(prompt_price) * 1000
        output_per_1k = float(completion_price) * 1000
    except (ValueError, TypeError):
        return None

    # Split provider/model
    parts = model_id.split("/", 1)
    if len(parts) != 2:
        return None
    provider, short_name = parts

    return {
        "provider": provider,
        "model_id": model_id,
        "short_name": short_name,
        "input_per_1k": input_per_1k,
        "output_per_1k": output_per_1k,
    }


def toml_escape(s):
    """Escape a string for use as a TOML key (quote if needed)."""
    if all(c.isalnum() or c in "-_." for c in s):
        return s
    return f'"{s}"'


def generate_toml(models, providers=None):
    """Generate the [plugins.config.model_costs] TOML section."""
    # Group by provider
    by_provider = {}
    for m in models:
        p = m["provider"]
        if providers and p not in providers:
            continue
        by_provider.setdefault(p, []).append(m)

    # Sort providers by preferred order, then alphabetically
    def provider_sort_key(p):
        try:
            return (PROVIDER_ORDER.index(p), p)
        except ValueError:
            return (len(PROVIDER_ORDER), p)

    lines = ["[plugins.config.model_costs]"]
    for provider in sorted(by_provider.keys(), key=provider_sort_key):
        group = sorted(by_provider[provider], key=lambda m: m["short_name"])
        lines.append(f"# {provider}")
        for m in group:
            key = toml_escape(m["short_name"])
            lines.append(
                f'{key} = {{ input = {m["input_per_1k"]:.6g}, '
                f'output = {m["output_per_1k"]:.6g} }}'
            )
    return "\n".join(lines)


def generate_full_config(model_costs_toml, port=8888):
    """Generate a complete example config file."""
    return f"""\
port = {port}

[paths]
"/v1/*" = ["https://api.openai.com"]
"/anthropic/*" = ["https://api.anthropic.com"]

# --- Rate Limiter ---
[[plugins]]
name = "rate_limiter"
enabled = true
[plugins.config]
tokens_per_minute = 100000
burst_tokens = 20000

# --- Cost Tracker ---
# Per-model pricing synced from OpenRouter.
# Re-generate with: python3 scripts/sync_model_prices.py
[[plugins]]
name = "cost_tracker"
enabled = true
[plugins.config]
budget_limit = 0.0
log_interval_secs = 60
default_cost_per_1k_input = 0.01
default_cost_per_1k_output = 0.03

{model_costs_toml}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Sync model pricing from OpenRouter for cost_tracker plugin"
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        help="Only include these providers (e.g. openai anthropic google)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Output a complete example config file instead of just model_costs",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List all available providers and exit",
    )
    args = parser.parse_args()

    print("Fetching models from OpenRouter...", file=sys.stderr)
    raw_models = fetch_models()

    models = []
    for entry in raw_models:
        parsed = parse_model(entry)
        if parsed and (parsed["input_per_1k"] > 0 or parsed["output_per_1k"] > 0):
            models.append(parsed)

    print(f"Found {len(models)} models with pricing", file=sys.stderr)

    if args.list_providers:
        providers = sorted(set(m["provider"] for m in models))
        for p in providers:
            count = sum(1 for m in models if m["provider"] == p)
            print(f"  {p} ({count} models)")
        return

    providers = set(args.providers) if args.providers else None
    toml_snippet = generate_toml(models, providers)

    if args.full:
        print(generate_full_config(toml_snippet))
    else:
        print(toml_snippet)

    included = sum(
        1 for m in models
        if not providers or m["provider"] in providers
    )
    print(f"\nGenerated pricing for {included} models", file=sys.stderr)


if __name__ == "__main__":
    main()
