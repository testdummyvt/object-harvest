# Agent Guidelines for object-harvest

## Overview
object-harvest is a Python 3.12+ tool for generating and processing object-related data:
- Prompt generation using LLMs (via OpenRouter)
- Image captioning using Moondream API
- VLM object detection with bounding boxes
- NDJSON structured output

All runtime code belongs under `obh/`. Treat every change as greenfield.

## Environment Setup
Always work through `uv`; never call `python`/`pip` directly.

```bash
uv venv
uv install --dev
```

Add packages with `uv add <package>` to keep `pyproject.toml` as the single source of truth.

## Build/Lint/Test Commands

### Linting and Formatting
```bash
uv run ruff check --fix .
```
Run this before every commit. Ruff is the only configured linter.

### Running Tests
```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_detect.py

# Run a single test function
uv run pytest tests/test_detect.py::test_vlm_task_with_mocked_dependencies

# Run with verbose output
uv run pytest -v
```

## Code Style Guidelines

### Imports
Order: standard library, third-party, local (obh.*). Use explicit imports, avoid `*`.

```python
import os
from typing import Optional, Dict, Any

import click
from tqdm import tqdm

from obh.utils import setup_llm_client, rate_limited_call
```

### Type Annotations
All public functions/classes require full type annotations. Use standard typing module types:
- `Optional[T]`, `List[T]`, `Dict[K, V]`, `Any`
- For classes, use `Mapping[str, Tuple[int, int]]` etc.
- `from __future__ import annotations` is not used (Python 3.12+)

```python
def process_image(img_path: str) -> Optional[Dict[str, Any]]:
    """Process an image and return metadata."""
    ...
```

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Private functions/modules: `_leading_underscore`

```python
MAX_SEED = np.iinfo(np.int32).max

class QwenImage:
    def __call__(self, prompt: str) -> Image.Image:
        ...
```

### Docstrings
Google-style with Args/Returns/Raises sections. Only add where business logic needs context.

```python
def validate_response(response: str) -> Dict[str, Any]:
    """Parse and validate LLM response.

    Args:
        response: Raw LLM response string.

    Returns:
        Validated dict with 'describe' and 'objects'.

    Raises:
        ValueError: If response cannot be validated.
    """
```

### Error Handling
- Use `ValueError` for validation errors with descriptive messages
- Try/except specific exceptions, avoid bare except
- Task functions return `int`: 0 for success, 1 for failure
- CLI functions raise `click.ClickException` on failure

```python
def setup_llm_client(base_url: str, api_key: Optional[str]) -> OpenAI:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not provided")
    return OpenAI(base_url=base_url, api_key=api_key)
```

### CLI Patterns
Use Click framework with decorators:
- Common options: define `add_common_llm_options` decorator
- Command groups: `@click.group()`, `@cli.command()`
- Args use `click.option()` with type hints
- Raise `click.ClickException` for CLI errors

```python
@click.command()
@click.option("--input", type=str, required=True)
@click.option("--output", type=str, required=True)
@add_common_llm_options
def vlm(input: str, output: str, model: str, ...) -> None:
    """Detect objects in images using VLM."""
    ...
```

## Project Layout
```
obh/
  __init__.py              # Intentionally empty; expose public entry points here
  generate.py              # CLI for generation tasks (prompt-gen, moondream-caption)
  detect.py                # CLI for detection tasks (vlm)
  utils/
    __init__.py            # Export shared utilities
    llm_utils.py           # LLM client, rate limiting, helpers
    moondream_utils.py     # Moondream API utilities
    validation.py           # Response validation
    prompts.py             # System prompts
tests/
  test_generate.py         # Tests for generate.py
  test_detect.py           # Tests for detect.py
```

## Development Workflow
1. Make changes and run `uv run ruff check --fix .` to fix formatting/linting
2. Run relevant tests: `uv run pytest tests/<path>`
3. Add tests for new features (tests/ mirrors obh/ structure)
4. Delete dead code instead of commenting out; update dependent tests in same change
5. Keep PRs narrowly scoped: one feature or refactor per PR
6. Update README.md if change alters usage expectations

## Quality Standards
- Public functions/classes: full type annotations + docstrings
- Internal helpers: type annotations, docstrings only if logic needs context
- Avoid `# type: ignore` without short justification + follow-up issue link
- Modules should be small and focused
- Use `concurrent.futures.ThreadPoolExecutor` with `tqdm` for progress on batch work
- Rate limiting: use `rate_limited_call` wrapper for LLM API calls
- NDJSON/JSONL format: one JSON object per line with `json.dumps() + "\n"`

## References
- `pyproject.toml` — dependency list, bump version for user-visible changes
- `.github/copilot-instructions.md` — additional context
- `README.md` — usage documentation, keep in sync with changes
