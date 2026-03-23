# Management API

The LLM gateway exposes an HTTP management API on the configured
`management_api_port`. It is the control-plane surface for operator status,
virtual keys, governance, safety, tools, and prompt-cache/runtime visibility.

## Auth

- If gateway auth is enabled, send `Authorization: Bearer <token>`.
- For local bootstrap flows, set `TRP_BOOTSTRAP_ADMIN_TOKEN` or
  `CreatePluginsOptions.bootstrap_admin_token` so an initial admin path exists.
- Runtime inference keys and management API bearer tokens are separate.

## Core status endpoints

- `GET /api/v1/status`
  - plugin enablement summary
  - budget limit and tracked-key counts where applicable
- `GET /api/v1/providers`
  - configured providers
  - tool protocol / managed-tool capability
  - prompt-cache protocol / request-control capability
  - configured models and per-provider timeout override
- `GET /api/v1/providers/health`
  - in-memory routing health snapshot
  - cooldown reason / remaining time
  - EWMA latency/error/timeout/rate-limit stats
  - adaptive penalty breakdown used by the current scorer
- `GET /api/v1/providers/failed`
  - current cooldown list with explicit failure reasons
- `DELETE /api/v1/providers/failed`
  - clears provider cooldown state
- `GET /api/v1/rate-limiter/status`
  - configured rate and burst plus tracked key count
- `GET /api/v1/tool-runtime/status`
  - webhook/web-search backends
  - MCP server reachability / discovery / recovery / budget status
  - operator state, derived health state, and recommended action for each MCP server
  - executor support and registered/enabled tool counts
- `POST /api/v1/tool-runtime/mcp/{server_name}/refresh`
  - forces MCP `initialize` + `tools/list`
  - refreshes cached session/discovery state for one server
- `POST /api/v1/tool-runtime/mcp/{server_name}/disable`
  - disables one MCP server in memory without restarting the gateway
  - new tool calls fail fast with an operator-facing disabled reason
- `POST /api/v1/tool-runtime/mcp/{server_name}/enable`
  - re-enables a previously disabled MCP server without restarting the gateway
- `DELETE /api/v1/tool-runtime/mcp/{server_name}/session`
  - sends MCP session termination when a cached session exists
  - clears local session state so the next tool call reinitializes cleanly
- `GET /api/v1/prompt-cache/status`
  - default Anthropic cache scope
  - per-provider prompt-cache protocol support

## Usage and pricing

- `GET /api/v1/cost/usage`
- `GET /api/v1/cost/usage/by-model`
- `DELETE /api/v1/cost/usage`
- `DELETE /api/v1/cost/usage/{key}`
- `GET /api/v1/cost/models`
- `PUT /api/v1/cost/models/{model}`
- `DELETE /api/v1/cost/models/{model}`

These endpoints back budget enforcement, model pricing updates, and operator
usage inspection.

## Logs and sessions

- `GET /api/v1/logs`
  - request log listing
  - supports `session_id` filtering when store-backed logs are enabled
- `GET /api/v1/sessions`
  - lists durable session records
  - supports `project_id`, `status`, `owner_id`, `updated_after_unix`, `limit`
  - also supports derived filters like `recovery_required`, `handoff_pending`,
    `cancel_pending`, and `owner_stale`
- `GET /api/v1/sessions/{session_id}`
  - durable session rollup plus aggregate summary
  - first/last request timestamps
  - latest request context (provider/model/prompt)
  - session-scoped token, cost, safety, semantic, and tool activity totals
  - persisted `status`, `state`, and `metadata` when set
- `GET /api/v1/sessions/{session_id}/events`
  - durable session event feed for orchestration
  - includes request-observed events as well as ownership/lifecycle mutations
  - supports `after_seq` and `limit`
- `GET /api/v1/sessions/{session_id}/wait`
  - long-polls the session event feed until a newer event arrives or timeout expires
  - supports `after_seq`, `timeout_secs`, and `limit`
- `PUT /api/v1/sessions/{session_id}`
  - upserts durable session `status`, `state`, and `metadata`
  - preserves existing aggregate request counters and latest-request context
- `POST /api/v1/sessions/{session_id}/claim`
  - claims a long-running session for a specific runtime owner
  - refreshes lease expiry and records ownership timestamps
- `POST /api/v1/sessions/{session_id}/release`
  - releases an active session claim
  - clears the persisted owner and lease
- `POST /api/v1/sessions/{session_id}/handoff`
  - records a pending transfer from the current owner to a target owner
  - keeps the current owner active until the target explicitly accepts
- `POST /api/v1/sessions/{session_id}/accept`
  - accepts a pending handoff for the target owner
  - rotates the active owner, refreshes lease state, and clears pending handoff metadata
- `POST /api/v1/sessions/{session_id}/takeover`
  - explicitly resumes ownership when a lease is stale or an operator/runtime
    needs to override an active owner
  - rejects active-owner conflicts unless `force=true` is supplied
- `POST /api/v1/sessions/{session_id}/cancel`
  - records a cancellation request with operator/runtime attribution
  - allows the runtime or the gateway reconciler to finalize cancellation later
- `POST /api/v1/sessions/{session_id}/reconcile`
  - applies stale-session reconciliation immediately for operator control
  - pauses sessions whose owner lease expired and finalizes pending cancels when
    no active owner remains
- `POST /api/v1/sessions/{session_id}/transition`
  - applies validated lifecycle transitions such as `active -> paused` or
    `active -> completed`
  - persists transition timestamp and optional operator reason
- `POST /api/v1/sessions/{session_id}/heartbeat`
  - refreshes session liveness for long-running work
  - can extend a persisted lease expiry and update state/metadata in place
- `GET /api/v1/sessions/{session_id}/logs`
  - raw request-log rows for a specific session

These endpoints are the current continuity surface for multi-request traffic.
Request logs remain the source events, and the gateway now persists a durable
session rollup/state record on top of them for faster reads and longer-lived
runtime continuity, including explicit ownership, cancellation intent,
lifecycle transitions, heartbeats, handoff state, lease-aware takeover, and
stale-session reconciliation. Session events now provide a durable orchestration
feed over both request activity and management-side session mutations, plus a
simple long-poll wait surface for runtimes that need to react to the next
change.

## Virtual keys and access

- `POST /api/v1/keys`
- `GET /api/v1/keys`
- `GET /api/v1/keys/{hash_prefix}`
- `PATCH /api/v1/keys/{hash_prefix}`
- `DELETE /api/v1/keys/{hash_prefix}`

The gateway uses virtual keys for project scoping, provider routing, budgets,
RPM/TPM overrides, and managed runtime features like tools and prompt cache.

Projects, principals, role bindings, and tokens are also managed through the
same API namespace:

- `GET/POST /api/v1/projects`
- `GET/POST /api/v1/principals`
- `GET/POST /api/v1/role-bindings`
- `GET /api/v1/tokens`

## Project governance resources

Project-scoped policy endpoints cover the main runtime controls:

- project policy
- routing rules
- safety policies
- semantic-safety policies and sync status
- project tools

These are exposed under `/api/v1/projects/{project_id}/...` and persist through
SQLite, Postgres, or MySQL when `store_url` is configured.

## Operational guidance

- Use `GET /api/v1/providers/health` when debugging routing or failover.
- Use `GET /api/v1/tool-runtime/status` when managed tools or MCP-backed tools
  are not behaving as expected.
- Use `POST /api/v1/tool-runtime/mcp/{server_name}/refresh` when an MCP server
  has changed inventory or you want to force a clean rediscovery.
- Use `DELETE /api/v1/tool-runtime/mcp/{server_name}/session` when a remote MCP
  session has gone stale and you want the next request to reinitialize it.
- Use `GET /api/v1/prompt-cache/status` plus `GET /api/v1/providers` to confirm
  which providers can accept gateway-managed cache controls.
- Keep management API auth enabled anywhere beyond local development.
