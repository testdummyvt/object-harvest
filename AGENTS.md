# AGENTS.md

## Prerequisites & Environment Setup

* **Python version requirement**:
  The project requires **Python ≥ 3.12** (i.e. 3.12 or newer).
* **Environment management tool**:
  All agents **must** use `uv` (the uv environment manager) for managing the Python environment (virtual environment, dependencies).

  * e.g. `uv init`, `uv venv`, `uv install`, `uv run ...`, etc.
  * Do *not* use `venv` or `pipenv` or `poetry` in isolation; `uv` must wrap or orchestrate all environment and execution.
* **Dependency installation**:
  After cloning, an agent should typically run something like:

  ```bash
  uv venv
  uv install --dev
  ```

  (Assuming your project’s `pyproject.toml` or equivalent supports a “dev” extras install.)
* **Activating / running commands**:
  Use `uv run <command>` to invoke test, lint, format, etc., so they run in the right environment.

## Code Style & Linting

We maintain **strict enforcement** of linting and style. The guiding tool is **ruff**.

* Every pull request **must** pass:

  ```bash
  uv run ruff check --fix .
  ```

  before merging (or as part of CI).
* The linter may autocorrect many issues; but remaining violations must be addressed by the author.
* Do **not** disable rules globally unless there is a very strong justification (and discussed with maintainers).
* Prefer idiomatic, clear Python style, consistent naming, and readability.

Additional style expectations:

* Modular code: keep functions/classes small and single-purpose.
* Appropriate docstrings (PEP-257 style) for public APIs.
* Follow PEP-8 naming and formatting conventions (as enforced by ruff).
* No trailing whitespace, correct indentation, line lengths (as configured), etc.

## Typing & Type Safety (`typing`)

We aim for high-quality type annotations. The following rules apply:

1. **Explicit types are preferred**

   * Annotate function arguments and return types unless in extremely trivial local lambdas.
   * Use `-> None`, `-> int`, `-> MyClass`, `-> Union[...]`, etc., rather than relying on inference.

2. **Allowed constructs**

   * `cast(...)` is okay when you know a type conversion but need to appease the type checker.
   * `assert ...` is acceptable when narrowing types in control flow (e.g. `assert x is not None`).
   * In **rare, simple cases**, an untyped argument (e.g. `def reward(state): ...`) may be allowed if it is trivial and internal; but strive to gradually add types.

3. **Forbidden / discouraged**

   * Avoid sprinkling `# type: ignore` unless there is **strong justification**, documented with a comment explaining *why*.
   * Do *not* suppress type errors silently or globally.
   * Unchecked dynamic typing should be the exception, not the norm.

4. **Type checking in CI**

   * In continuous integration (CI), we should run `mypy` (or equivalent) to enforce that there are no new typing errors.
   * PRs which introduce type errors should be blocked.

## Testing

Testing is essential for project correctness, stability, and future refactoring.

* Use **pytest**, with discovery under `tests/`.
  Command:

  ```bash
  uv run pytest
  ```
* **Tests should be:**

  * **Simple and deterministic** — avoid randomness or external dependencies.
  * **Unit tests first** — smaller, isolated functions and classes.
  * **Coverage for new behavior** — each new public function or behavior must have appropriate test coverage.
* **Test organization**

  * `tests/` directory mirrors some module structure of the code, where appropriate.
  * Use fixtures only when beneficial, avoid over-complex test machinery.
* **No flaky or timing-dependent tests**

  * If a test sometimes fails due to timing, rewrites or stabilization is required.
  * External I/O or network calls should be mocked or isolated.

## Pull Request / Review Workflow

To keep the codebase clean and manageable:

1. **One logical change per PR**

   * Each PR should focus on a single feature, bugfix, or refactor.
   * Do not bundle multiple unrelated changes in one PR.

2. **Small diffs, incremental improvement**

   * Break large changes into smaller, reviewable chunks when possible.
   * It’s easier to review and reason about small changes.

3. **Backward compatibility policy**

   * Backward compatibility is *desirable*, but **not mandatory** if supporting it would impose excessive maintenance burden (e.g. building in lots of legacy code paths).
   * If removing or changing a public API, update relevant tests and document the change in changelog / release notes.

4. **Review checks**
   Before merging:

   * Ensure `uv run ruff check --fix .` yields no violations.
   * Ensure `uv run pytest` passes.
   * Review type annotations; ensure no `# type: ignore` remains un-justified.
   * Confirm the change is focused and does not unduly affect unrelated code.

5. **Commit messages**

   * Use descriptive commit messages, e.g. `Add feature X`, `Fix bug in Y`, `Refactor Z for clarity`.
   * If a change is nontrivial, include rationale or linking to issue/PR discussion.

## Scope & Change Philosophy

* The project should evolve **incrementally** and **managably**.
* Changes should be **as focused as possible**, with small, self-contained patches.
* Maintain readability, maintainability, and clarity as guiding principles.
* Avoid overengineering; do not add features or abstractions prematurely (“You ain’t gonna need it”).
* If backward compatibility (BC) is expensive or complicates code, it may be acceptable to break compatibility — but document it clearly (in changelog or release notes).

## Dead Code & Refactoring

* **Delete dead code outright** — do *not* keep dead branches behind flags or commented out code.
* When functionality is changed or removed, **update or remove corresponding tests** immediately.
* Refactor code as necessary to reduce duplication or improve clarity; but always accompany refactor with tests to preserve behavior.
* Avoid speculative abstractions; refactor when you see a concrete need or duplication.

## Agent Responsibilities & Checklist

Here’s a checklist each agent (contributor) should keep in mind when working on *object-harvest*:

| Area                   | Agent Responsibility                                                         |
| ---------------------- | ---------------------------------------------------------------------------- |
| Environment tool       | Use `uv` (never raw `venv`, pipenv, etc.)                                    |
| Python version         | Ensure the Python interpreter is ≥ 3.12                                      |
| Activate environment   | Run `source .venv/bin/activate` (or equivalent) before using the environment |
| Linting                | Run `uv run ruff check --fix .` locally before pushing                       |
| Typing                 | Add explicit type annotations; avoid unjustified `# type: ignore`            |
| Testing                | Add or update tests under `tests/`; ensure `uv run pytest` passes            |
| PR Size                | Keep changes focused; one logical change per PR                              |
| Backward Compatibility | Consider it, but not at cost of code clarity                                 |
| Dead Code              | Remove rather than hide; update tests accordingly                            |
| Commit / PR            | Write descriptive commits; keep the review size manageable                   |
