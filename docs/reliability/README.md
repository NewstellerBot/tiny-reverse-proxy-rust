# Reliability Program

This package turns competitor research into a concrete reliability program for
this repo.

It is intentionally scenario-based, not feature-count-based. The question is
not "does Tiny Proxy have the same features as another gateway?" The question
is "do we cover the same reliability-critical behaviors with tests, benchmarks,
smokes, soak runs, and release gates?"

Use these documents together:

- [Reference Set](reference-set.md)
- [Scenario Taxonomy](scenario-taxonomy.md)
- [Parity Matrix](parity-matrix.md)
- [Current Coverage](current-coverage.md)
- [Missing Suites](missing-suites.md)
- [Release Gates](release-gates.md)

Working rules for this package:

- Use LiteLLM and Bifrost as open-source reference projects.
- Use Cloudflare AI Gateway, OpenRouter, Portkey, Helicone, and Langfuse as
  behavior and operator-contract references only.
- Do not create parity rows for hosted-only UI features, billing/admin
  features, or vendor-specific hosted integrations that do not affect gateway
  correctness or reliability.
- Prefer black-box scenario coverage over test-by-test imitation.

This package is the source of truth for the next reliability-testing wave:

- soak and long-run degradation detection
- retry/fallback classification
- multi-node shared-store correctness
- compatibility goldens
- observability contract validation
