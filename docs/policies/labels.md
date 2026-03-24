# GitHub Labels

The repository uses labels as operational signals, not decoration.

## Required labels

- `regression`
- `release-blocker`
- `needs-mre`
- `needs-design`
- `preview`
- `provider-specific`
- `platform-specific`
- `store-specific`
- `performance`
- `blocked-upstream`
- `security`
- `reliability`
- `docs`
- `good-first-issue`

## How labels are used

- `regression`: behavior worked before and now fails
- `release-blocker`: must be fixed before the next release
- `needs-mre`: issue needs a minimal reproducible example
- `needs-design`: feature or architectural change needs up-front design agreement
- `preview`: issue or PR touches preview-only behavior
- `provider-specific` / `platform-specific` / `store-specific`: triage scope
- `performance`: hot-path or benchmark-sensitive work
- `blocked-upstream`: cannot complete locally without upstream fix

Labels are managed from `.github/labels.yml`.
