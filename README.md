# Object Harvest

Extract image descriptions and object lists from images using Vision-Language Models (VLMs). Now includes open-vocabulary detection and text synthesis. Inputs can be a folder of images or a text file of paths/URLs. Outputs are one JSON file per image, saved under a unique run folder.

## Features

- Inputs: folder of images or text list file (paths and/or URLs)
- Outputs: one JSON per image under a unique run directory (run-YYYYMMDD-HHMMSS-<id>)
- OpenAI-compatible API client (set `--api-base` for OpenRouter/others)
- Default model: `qwen/qwen2.5-vl-72b-instruct` (override via `--model` or `OBJH_MODEL`)
- Concurrency via threads + a shared RPM rate limiter (`--rpm`)
- Subcommands:
  - `describe` — image caption + objects list (existing behavior)
  - `detect` — open-vocabulary detection (GroundingDINO/LLMDet; WIP skeleton)
  - `synthesis` — generate a one-line description from a list of objects

## Installation

This project uses a `src/` layout and is installable (setuptools backend). Prefer `uv` for environment management.

- Option A — Install CLI with uv (recommended):

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

# Now the console script is on PATH
object-harvest --help
object-harvest describe --help
object-harvest detect --help
object-harvest synthesis --help
```

- Option B — Run without installing (uv run):

```bash
uv sync
uv run -m object_harvest.cli --help
uv run -m object_harvest.cli describe --help
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

### Describe (caption + objects)

Folder of images (installed CLI):

```bash
object-harvest describe \
  --input ./images \
  --out ./out \
  --model qwen/qwen2.5-vl-72b-instruct \
  --rpm 30 \
  --max-workers 16
```

Text list file (installed CLI):

```bash
object-harvest describe \
  --input ./list.txt \
  --out ./out
```

Alternative (no install):

```bash
uv run --env-file .env -m object_harvest.cli describe --input ./images --out ./out
```

Flags (describe):

- `--input <folder|list.txt>`: Folder of images or a text file with one path/URL per line
- `--out <folder or out/run-*>`: Output directory where a run folder will be created (or an existing run-* dir when `--resume`)
- `--model <name>`: Model name (default `qwen/qwen2.5-vl-72b-instruct`)
- `--api-base <url>`: OpenAI-compatible base URL (optional)
- `--rpm <N>`: Requests per minute throttle shared across threads (0 = unlimited)
- `--max-workers <N>`: Thread pool size
- `--batch <N>`: Process only the first N items (for quick tests)
- `--resume`: Resume a previous run by writing into an existing run-* directory and only processing images without a JSON output yet. If `--out` points to a run-* folder, it's used directly; otherwise the latest run-* under `--out` is selected.

### Resume only missing outputs

Continue a previous run and only process images that don't yet have a JSON file:

```bash
object-harvest describe --input ./images --out ./out --resume
```

You can also target a specific run directory:

```bash
object-harvest describe --input ./images --out ./out/run-20250822-104455-ab12cd34 --resume

### Detect (open-vocabulary detection)

Use detections based on a list of objects or reuse the objects produced by a previous describe run:

```bash
object-harvest detect \
  --input ./images \
  --out ./detections \
  --backend gdino \
  --hf-model IDEAS-LAB/grounding-dino-base \
  --from-describe ./out/run-20250822-104455-ab12cd34 \
  --threshold 0.25 \
  --max-workers 8 \
  --resume
```

Notes:
- Backend implementation is a WIP skeleton; install `transformers` and `torch` and set `--hf-model` to enable.
- You can also provide `--objects-file` or `--objects` instead of `--from-describe`.

### Synthesis (generate a one-line description from object names)

Generate a description that includes N objects from a provided list:

```bash
object-harvest synthesis \
  --objects-file ./objects.txt \
  --n 6 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --out ./synthesis.json
```
```

## Output

The CLI writes a unique run directory under `--out`, e.g. `out/run-20250822-104455-ab12cd34/`, with one JSON per image. Each file conforms to:

```json
{
  "image": "file_or_url",
  "description": "A concise one-line scene description naming all objects...",
  "objects": ["object1", "object2", "..."]
}
```

Example filenames:

- For local paths: `cat_photo.json` (derived from basename)
- For URLs: `photo.jpg.json` (derived from URL path)

## Notes

- Describe is production-ready; detect is a scaffold to integrate models like GroundingDINO/LLMDet; synthesis uses the LLM API.
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
- Detection backends: ensure `transformers` and `torch` are installed and the chosen HF model is available.
- Module not found: ensure `uv run -m object_harvest.cli` or `PYTHONPATH=src` when using `python -m ...`.
 