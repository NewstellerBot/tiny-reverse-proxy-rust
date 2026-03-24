# Multi-node correctness harness

This harness exercises the documented multi-node deployment model against three local
`tiny-reverse-proxy` processes and a shared Postgres store.

It is intentionally restart-oriented today:
- shared control-plane state is authoritative in Postgres
- node-local caches are advisory
- cross-node convergence is verified via restart/reload boundaries, not by assuming background
  replication that does not exist yet

Scenarios covered:
- project + key created through node A become usable on nodes B and C after restart-based reload
- one-at-a-time rolling restarts keep traffic flowing to ready nodes
- readiness drain removes a node before the client balancer sends more traffic to it
- project config apply/rollback converges after restart-based reload
- node-local cache loss changes warm-path behavior only; correctness stays intact

Run locally with:

```bash
docker compose -f tests/reliability/docker-compose.multinode.yml up -d postgres
uv run python tests/reliability/multinode/run.py
```

Useful flags:
- `--keep-running` leaves Postgres and node processes up for manual inspection
- `--binary target/debug/tiny-reverse-proxy` uses an already-built binary
- `--store-url ...` points at an existing shared Postgres instance

The script builds a temporary workspace under `tests/reliability/multinode/.tmp/`.
