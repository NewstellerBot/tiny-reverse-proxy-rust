# Change Management

## Design discussion required

Open a design discussion before implementation for:

- a new provider family
- a new endpoint family
- a new control-plane mutation model
- a new preview feature
- a hot-path architecture change

## Pull request rules

Every PR should state:

- change category
- regression test added, or why it is not needed
- benchmark note, or why the change is not hot-path sensitive
- rollback note for config/runtime-affecting changes
- preview/stability impact

## Bug fixes

- Confirmed bugs require a regression test before close.
- Regressions get the `regression` label.
- Release-impacting regressions get the `release-blocker` label.

## Scope discipline

Avoid broad rewrites without a direct user-facing win. Changes should default to the smallest slice
that improves correctness, reliability, or operability.
