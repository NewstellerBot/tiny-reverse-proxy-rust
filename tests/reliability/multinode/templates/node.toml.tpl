port = {port}
management_api_port = {management_api_port}
management_api_token = "$TRP_BOOTSTRAP_ADMIN_TOKEN"
store_url = "{store_url}"
allow_direct_provider_keys = false

[paths]
"/v1/*" = ["http://127.0.0.1:{primary_upstream_port}"]

[reliability]
max_inflight_requests = 128
brownout_inflight_requests = 96
retry_budget_per_request = 2

[[providers]]
name = "openai"
api_key = "sk-mock-openai"
base_url = "http://127.0.0.1:{primary_upstream_port}"
models = ["gpt-4o"]
api_key_header = "authorization"
family = "openai"

[providers.surfaces]
tools = "openai"
responses = "openai_compatible"
