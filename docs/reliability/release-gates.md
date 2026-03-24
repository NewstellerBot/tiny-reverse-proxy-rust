# Release Gates

This document places each reliability validation mode into a specific lane.

The goal is to avoid re-deciding where new suites belong every time we add one.

## Lane map

| Lane | Where it lives | What belongs there |
|---|---|---|
| PR CI | existing `.github/workflows/ci.yml` | unit tests, deterministic integration tests, compile-level benchmark smoke, fast regression coverage |
| Release-branch CI | new future `.github/workflows/release-branch-validation.yml` | deterministic fault-injection integration suites, compatibility goldens, observability contract tests, multi-node correctness subset |
| Release workflow hard gates | existing `.github/workflows/release.yml` and `.github/workflows/live-openai-responses-smoke.yml` | deterministic release validation plus live gateway `/v1/responses` smoke |
| Maintainer-triggered validation | new future `.github/workflows/release-soak.yml` or equivalent manual workflow | 2-hour RC soak, longer stable-candidate soak, competitor black-box spot checks |

## Default placement by validation mode

| Validation mode | Default lane |
|---|---|
| `unit` | PR CI |
| `integration` | PR CI unless the suite is slow or infrastructure-heavy; otherwise Release-branch CI |
| `live_smoke` | Release workflow hard gates |
| `benchmark` | PR compile/smoke only if lightweight; otherwise maintainer-triggered |
| `soak` | maintainer-triggered |
| `manual_release_check` | maintainer-triggered |

## Explicit decisions for the next wave

### Stay in PR CI

- existing workspace unit and integration suites
- benchmark harness compile/sanity checks only
- fast retry/fallback unit coverage where it does not require shared
  infrastructure

### Move to release-branch CI once implemented

- retry-classification integration suite
- compatibility goldens suite
- observability contract suite
- multi-node correctness subset
- deterministic fault-injection scenarios

### Stay as release hard gates

- existing deterministic release validation in `release.yml`
- existing live gateway `/v1/responses` smoke in
  `live-openai-responses-smoke.yml`

### Stay maintainer-triggered

- 2-hour RC soak
- 12-hour stable-candidate soak
- competitor black-box spot checks against hosted services where cost or rate
  limits matter

## Non-negotiable rules

- Do not put secret-backed live provider tests on normal PRs.
- Do not put multi-hour soak runs on ordinary PRs.
- Do put retry/fallback classification and multi-node correctness into a
  release-branch validation lane once implemented.
- Do not create a release tag unless the release workflow and live smoke both
  pass.
