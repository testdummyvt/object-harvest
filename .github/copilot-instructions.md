<!-- Copilot/AI agent instructions for contributing to object-harvest -->

Overview
--------
- `object-harvest` is an empty Python 3.12+ scaffold; all runtime code belongs under `obh/`.
- Existing guidance lives in `AGENTS.md`; mirror its expectations unless a ticket states otherwise.
- Treat every change as greenfield: you will usually be defining fresh modules, tests, and docs together.

Environment & Tooling
---------------------
- Always work through `uv`; never call `python`/`pip` directly.
- Typical setup when starting fresh on this repo:
  - `uv venv`
  - `uv install --dev`
  - `uv run ruff check --fix .`
- Add packages with `uv add <package>` so `pyproject.toml` stays the single source of truth.

Repo Layout & Conventions
-------------------------
- `obh/__init__.py` is intentionally empty; use it to expose public entry points as packages evolve.
- Create new modules under `obh/` and mirror them under `tests/`; the test tree does not exist yet, so create directories explicitly.
- Keep modules small and typed. Public functions/classes require full annotations and docstrings describing intent.
- Delete dead code instead of commenting it out; update or drop dependent tests in the same change.

Development Workflow
--------------------
- Use `uv run ruff check --fix .` before any commit to satisfy formatting and linting (`ruff` is the only configured tool today).
- Run targeted tests with `uv run pytest tests/<path>` as you add coverage; there are currently no baseline tests, so new features must include them.
- When exploring, prefer writing thin spike scripts inside `tests/` or scratch modules and removing them before finalizing.

Quality Bar
-----------
- Add docstrings and reasoning comments sparingly—only where business logic needs context.
- Avoid `# type: ignore` unless accompanied by a short justification and link to a follow-up issue.
- Keep pull requests narrowly scoped: one feature or refactor per set of changes; update `README.md` if the change alters usage expectations.

Reference Material
------------------
- `AGENTS.md` — canonical policy for environment, typing, and review checklist.
- `pyproject.toml` — authoritative dependency list; bump version when shipping user-visible behavior.
- `README.md` — currently empty; populate it when you introduce the first user-facing feature.
