# Versioning

## Public surface

The project treats these as the long-term public surface:

- request/response behavior of the proxy itself
- management API routes and response shapes
- top-level configuration schema after GA

Public changes should be called out in release notes and should not change casually.

## Faster-moving surface

These may evolve faster, especially before GA:

- crate-internal Rust APIs
- store schema and migration layout before GA
- preview features
- advanced provider-specific translation behavior

## MSRV

- The project tracks stable Rust.
- MSRV bumps must be called out in release notes.
- CI should always build on stable Rust.

## Compatibility posture before GA

The project is still pre-GA. That means it is allowed to tighten or redesign surfaces, but any
change that affects operators or downstream automation should still be documented as if it were a
compatibility change.
