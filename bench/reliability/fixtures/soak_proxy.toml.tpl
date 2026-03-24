port = {port}
management_api_port = {management_api_port}
management_api_token = "$TRP_BOOTSTRAP_ADMIN_TOKEN"
metrics_port = {metrics_port}
store_url = "sqlite:///{sqlite_path}"
allow_direct_provider_keys = false

[paths]
"/v1/*" = ["http://127.0.0.1:{mock_primary_port}"]

[reliability]
max_inflight_requests = {proxy_max_inflight_requests}
brownout_inflight_requests = {proxy_brownout_inflight_requests}
retry_budget_per_request = 2

[[providers]]
name = "mock-primary"
api_key = "sk-mock-primary"
base_url = "http://127.0.0.1:{mock_primary_port}"
models = ["gpt-4o", "gpt-4.1-mini"]
api_key_header = "authorization"
family = "openai"

[providers.surfaces]
tools = "openai"
responses = "openai_compatible"

[[providers]]
name = "mock-fallback"
api_key = "sk-mock-fallback"
base_url = "http://127.0.0.1:{mock_fallback_port}"
models = ["gpt-4o", "gpt-4.1-mini"]
api_key_header = "authorization"
family = "openai"

[providers.surfaces]
tools = "openai"
responses = "openai_compatible"

[[plugins]]
name = "provider_failover"
enabled = true

[plugins.config]
cooldown_secs = 30

[[plugins.config.providers]]
name = "mock-primary"
pattern = "http://127.0.0.1:{mock_primary_port}"

[[plugins.config.providers]]
name = "mock-fallback"
pattern = "http://127.0.0.1:{mock_fallback_port}"
{live_provider_block}
