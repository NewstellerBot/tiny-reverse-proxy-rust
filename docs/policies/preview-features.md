# Preview Features

Preview features are intentionally shipped, but they are not part of the stable support contract.

## Rules

- Preview features must be explicitly documented.
- Preview features must be visible in management/status output.
- High-risk operator-facing previews must require explicit opt-in.
- Preview features graduate only after docs, tests, and at least one release cycle of evidence.

## Current preview registry

- `responses_composed_streaming`
- `control_plane_import`
- `provider_surface_translations`

## Enforcement

- Hard-gated:
  - `responses_composed_streaming`
  - `control_plane_import`
- Visibility-only in v1:
  - `provider_surface_translations`

## How to enable

In application config, set a top-level `preview_features` array:

```toml
preview_features = ["responses_composed_streaming", "control_plane_import"]
```

When constructing the gateway programmatically, the same names may be supplied through LLM gateway
plugin config so tests and embedded callers can opt in without a full top-level config parse.

## Graduation criteria

A preview feature should not be promoted until:

- the wire/config shape is documented
- black-box coverage exists
- release notes explain the behavior
- no open release-blocking regressions remain for that feature
