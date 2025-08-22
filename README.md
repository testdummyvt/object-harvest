# Object Harvest

Extract image descriptions and object lists from images using Vision-Language Models (VLMs). Input can be a folder of images or a text file of paths/URLs. Output is one JSON file per image, saved under a unique run folder.

## Features

- Inputs: folder of images or text list file (paths and/or URLs)
- Outputs: one JSON per image under a unique run directory (run-YYYYMMDD-HHMMSS-<id>)
- OpenAI-compatible API client (set `--api-base` for OpenRouter/others)
- Default model: `qwen/qwen2.5-vl-72b-instruct` (override via `--model` or `OBJH_MODEL`)
- Concurrency via threads + a shared RPM rate limiter (`--rpm`)

## Installation

This project uses a `src/` layout and is installable (setuptools backend). Prefer `uv` for environment management.

- Option A — Install CLI with uv (recommended):

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

# Now the console script is on PATH
object-harvest --help
```

- Option B — Run without installing (uv run):

```bash
uv sync
uv run -m object_harvest.cli --help
```

- Option C — Python venv (no uv):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
object-harvest --help
```

## Configuration

Copy `.env.example` → `.env` and set:

- `OBJH_API_KEY` (required)
- `OBJH_API_BASE` (optional; set for OpenRouter or self-hosted gateways)
- `OBJH_MODEL` (optional; defaults to `qwen/qwen2.5-vl-72b-instruct`)
- `OBJH_RPM` (optional; default 0 = unlimited)

## Usage

Folder of images (installed CLI):

```bash
object-harvest \
  --input ./images \
  --out ./out \
  --model qwen/qwen2.5-vl-72b-instruct \
  --rpm 30 \
  --max-workers 16
```

Text list file (installed CLI):

```bash
object-harvest \
  --input ./list.txt \
  --out ./out
```

Alternative (no install):

```bash
uv run --env-file .env -m object_harvest.cli --input ./images --out ./out
```

Flags:

- `--input <folder|list.txt>`: Folder of images or a text file with one path/URL per line
- `--out <folder>`: Output directory where a run folder will be created
- `--model <name>`: Model name (default `qwen/qwen2.5-vl-72b-instruct`)
- `--api-base <url>`: OpenAI-compatible base URL (optional)
- `--rpm <N>`: Requests per minute throttle shared across threads (0 = unlimited)
- `--max-workers <N>`: Thread pool size
- `--batch <N>`: Process only the first N items (for quick tests)

## Output

The CLI writes a unique run directory under `--out`, e.g. `out/run-20250822-104455-ab12cd34/`, with one JSON per image. Each file conforms to:

```json
{
  "image": "file_or_url",
  "description": "A concise one-line scene description naming all objects...",
  "objects": ["object1", "object2", "..."],
  "bboxes": {}
}
```

Example filenames:

- For local paths: `cat_photo.json` (derived from basename)
- For URLs: `photo.jpg.json` (derived from URL path)

## Notes

- Currently only description and object list are generated; `bboxes` is an empty object placeholder.
- A single OpenAI client is shared across threads. Use `--rpm` to avoid provider rate limits.
- Supported image types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`.

## Development

- Code lives in `src/object_harvest/`.
  - `cli.py`: argument parsing and orchestration
  - `reader.py`: iterates inputs (folder or list file)
  - `vlm.py`: OpenAI-compatible client and prompt logic
  - `writer.py`: writes per-image JSON files to a unique run directory
  - `logging.py`: emoji-prefixed logger (use `get_logger(__name__)`)

## Troubleshooting

- 401/403 errors: verify `OBJH_API_KEY` and `OBJH_API_BASE`.
- 429 errors: reduce `--max-workers` and/or set `--rpm`.
- Module not found: ensure `uv run -m object_harvest.cli` or `PYTHONPATH=src` when using `python -m ...`.
