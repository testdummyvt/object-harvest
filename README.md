# Object Harvest

Extract object suggestions from images using Vision-Language Models (VLMs), perform open-vocabulary detection, and synthesize descriptions. Inputs can be a folder of images or a text file of paths/URLs. Describe now writes one NDJSON file per image (one JSON object per line), saved under a unique run folder.

## Features

- Inputs: folder of images or text list file (paths and/or URLs)
- Outputs: one file per image under a unique run directory (run-YYYYMMDD-HHMMSS-<id>)
  - Describe: .ndjson (one line per object: {"object": "short description"})
  - Detect: .json (detections array with XYXY pixel bboxes)
- OpenAI-compatible API client (set `--api-base` for OpenRouter/others)
- Default model: `qwen/qwen2.5-vl-72b-instruct` (override via `--model` or `OBJH_MODEL`)
- Concurrency via threads + a shared RPM rate limiter (`--rpm`)
- Subcommands:
  - `describe` — suggest objects as NDJSON lines (includes people when present)
  - `detect` — open-vocabulary detection via GroundingDINO or VLM-backed detection
  - `synthesis` — generate one-line descriptions from a list of objects

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

### Describe (object suggestions as NDJSON)

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
- `--resume`: Resume a previous run by writing into an existing run-* directory and only processing images without an NDJSON/JSON output yet. If `--out` points to a run-* folder, it's used directly; otherwise the latest run-* under `--out` is selected.

### Resume only missing outputs

Continue a previous run and only process images that don't yet have an output file:

```bash
object-harvest describe --input ./images --out ./out --resume
```

You can also target a specific run directory:

```bash
object-harvest describe --input ./images --out ./out/run-20250822-104455-ab12cd34 --resume

### Detect (open-vocabulary detection)

Status: W.I.P — detection is not functional yet. The CLI and output schema are in place; implementations are pending.

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
- Backends: `gdino` (GroundingDINO via transformers) and `vlm` (VLM-backed detection). For `gdino`, install `transformers` and `torch` and set `--hf-model`. You can optionally pass `--text` to use a free-form description prompt.
- Objects can come from a previous describe run (`--from-describe`) or via `--objects` (either a file path or a comma-separated list).

### Synthesis (generate one-line descriptions from object names)

Generate a description that includes N objects from a provided list:

```bash
object-harvest synthesis \
  --objects ./objects.txt \
  --num-objects 6 \
  --count 10 \
  --rpm 60 \
  --max-workers 8 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --out ./synthesis.jsonl

Flags (synthesis):
- `--objects <file|comma,list>` unified input (file path or comma-separated list)
- `--num-objects` (alias `--n`) number of objects to include per description (random sampled)
- `--count` number of samples to generate; with `--out` ending in .jsonl, outputs NDJSON; otherwise JSON/array
- `--rpm`, `--max-workers` for throughput control
- `--out` optional; if omitted, prints to stdout
```
```

## Output

The CLI writes a unique run directory under `--out`, e.g. `out/run-20250822-104455-ab12cd34/`.

Describe (.ndjson per image):
```
{"bicycle": "a blue road bike leaning against a brick wall"}
{"person": "a man wearing a red jacket walking beside the bike"}
{"backpack": "black backpack with a water bottle in side pocket"}
```

Detect (.json per image):
```json
{
  "image": "file_or_url",
  "detections": [
    {"label": "person", "score": 0.91, "bbox": {"xmin": 10.0, "ymin": 20.0, "xmax": 120.0, "ymax": 240.0}},
    {"label": "bicycle", "score": 0.88, "bbox": {"xmin": 40.0, "ymin": 60.0, "xmax": 300.0, "ymax": 220.0}}
  ]
}
```

## Notes

- Describe is production-ready; detection is W.I.P and does not work as of now (implementations pending). Synthesis uses the LLM API.
- A single OpenAI client is shared across threads. Use `--rpm` to avoid provider rate limits.
- Supported image types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`.

## Development

- Code lives in `src/object_harvest/`.
  - `cli.py`: argument parsing and orchestration
  - `reader.py`: iterates inputs (folder or list file)
  - `vlm.py`: OpenAI-compatible client and prompt logic (NDJSON for describe)
  - `writer.py`: writes per-image JSON files; also supports raw NDJSON via `write_text`
  - `logging.py`: emoji-prefixed logger (use `get_logger(__name__)`)

## Troubleshooting

- 401/403 errors: verify `OBJH_API_KEY` and `OBJH_API_BASE`.
- 429 errors: reduce `--max-workers` and/or set `--rpm`.
- Detection backends: ensure `transformers` and `torch` are installed and the chosen HF model is available.
- Module not found: ensure `uv run -m object_harvest.cli` or `PYTHONPATH=src` when using `python -m ...`.
 