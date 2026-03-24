# Deployment Topology

## Development topology

Single-node local deployment is supported for development, tests, and operator exploration.

## Recommended topology

The serious deployment model is:

- multiple proxy instances
- load balancer in front
- shared store for control-plane state
- isolated management API
- readiness/liveness integrated with rollout and drain behavior

## Unsupported topology

Do not rely on:

- node-local memory as authoritative control-plane state
- public management API exposure without protection
- multi-node deployments without a shared store

## Cache posture

Node-local caches are advisory. They may improve latency or provider routing, but they are not a
substitute for shared, authoritative state.
