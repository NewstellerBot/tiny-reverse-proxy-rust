# Release Policy

## Release branches

- Create `release/<major>.<minor>` from the default branch.
- Stabilization fixes are cherry-picked into the release branch.
- Do not cut releases directly from the default branch.

## Release candidates

- Run the existing GitHub Actions `Release` workflow from the release branch.
- Use prerelease versions such as `1.4.0-rc.1`.
- Treat RCs as the soak period for final release candidates.

## Final release

- Cut the final release from the same release branch after the RC soak period.
- Releases are created only by the GitHub Actions workflow.
- Do not create tags manually and do not publish from the GitHub UI.

## Hard gates

The release workflow must pass:

- deterministic release validation
- live gateway `/v1/responses` smoke
- protected environment approval for live secrets

## Release blockers

Do not release while any of these remain open:

- `release-blocker` issues
- `regression` issues targeted at the release
- Tier 1 CI failures
- preview/stable contract violations
