# Architecture and Reliability Findings (2026-03-04)

This file tracks issues identified in the code review and their remediation status.

## Critical

- [x] PROXY protocol parsing can consume request bytes and mishandle v2 fragmented headers.
  - Files: `crates/proxy-core/src/proxy_protocol.rs`, `crates/tiny-reverse-proxy/src/main.rs`
- [x] Plugin-selected upstream (`selected_upstream`) is not consistently honored by proxy forwarding.
  - Files: `crates/proxy-core/src/handlers/proxy.rs`, `crates/plugin-llm-gateway/src/virtual_keys.rs`

## High

- [x] Cache key ignores query string, causing collisions/poisoning between `?` variants.
  - File: `crates/proxy-core/src/handlers/proxy.rs`
- [x] Cache write path can replace errored upstream body with empty response.
  - File: `crates/proxy-core/src/handlers/proxy.rs`
- [x] Caching is not applied in the single-upstream single-attempt path.
  - File: `crates/proxy-core/src/handlers/proxy.rs`
- [x] Stateful LB strategies are recreated per request (`least-connections`, `weighted-round-robin`).
  - Files: `crates/proxy-core/src/handlers/proxy.rs`, `crates/proxy-core/src/load_balancer.rs`
- [x] Per-virtual-key RPM override creates a new bucket per request (no real limiting).
  - File: `crates/plugin-llm-gateway/src/rate_limiter.rs`
- [x] Management API is externally exposed and unauthenticated by default.
  - Files: `crates/plugin-llm-gateway/src/management_server.rs`, `crates/tiny-reverse-proxy/src/main.rs`

## Medium

- [x] `header_read_timeout_secs` is configured but not enforced.
  - File: `crates/tiny-reverse-proxy/src/main.rs`
- [x] `Accept-Encoding` negotiation ignores q-values.
  - File: `crates/proxy-core/src/compression.rs`
- [x] Config parsing silently coerces invalid integers and can silently drop providers when env vars are missing.
  - File: `crates/proxy-core/src/config.rs`
- [x] SIGHUP reload updates only router, leaving other settings stale.
  - File: `crates/tiny-reverse-proxy/src/main.rs`

## Notes

- All issues above have implemented fixes as of 2026-03-04.
- Regression tests were added for fixed behavior where feasible.
