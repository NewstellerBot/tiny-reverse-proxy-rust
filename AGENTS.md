# Repository Agent Instructions

## Python Tooling

- Use `uv` for Python-related work in this repository.
- Prefer `uv run python ...` instead of `python ...` or `python3 ...`.
- Prefer `uv run` for Python-based scripts, tests, and one-off tooling.
- Prefer `uv pip ...` instead of `pip ...` or `pip3 ...` when package installation is needed.
- If a Python command would normally use `python -m ...`, run it as `uv run python -m ...`.
- Only fall back to system Python tooling if `uv` is unavailable or a command is incompatible with `uv`; if that happens, say so explicitly.
