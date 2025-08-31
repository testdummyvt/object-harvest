# AI agent guide for object-harvest

Purpose: Extract NDJSON object suggestions from images using Vision-Language Models (VLMs), perform open-vocabulary detection (gdino or VLM-backed), and synthesize one-line descriptions from object lists—writing one file per image.

## Snapshot
- Python 3.12, src-layout: `src/object_harvest/`
- CLI present: `object-harvest` (entry → `object_harvest.cli:main`)
- Deps: `openai`, `pillow`, `tqdm`, `python-dotenv` (see `pyproject.toml`); `uv.lock` for env sync
- Config via `.env` (auto-loaded): `OBJH_API_KEY`, `OBJH_API_BASE`, `OBJH_MODEL`, `OBJH_RPM`

## Modules and roles
- `logging.py` — emoji logger. Use `get_logger(__name__)` once per module (idempotent handlers).
- `cli.py` — subcommands:
  - `describe` — flags: `--input`, `--out`, `--model`, `--api-base`, `--max-workers`, `--rpm`, `--batch`, `--resume`.
    - Emits one `.ndjson` per image: each line is `{ "object": "short description" }`. Include people when present.
  - `detect` — flags: `--input`, `--out`, `--backend {gdino,vlm}`, `--hf-model` (gdino), `--model`/`--api-base` (vlm), `--from-describe` | `--objects`, `--text`, `--threshold`, `--max-workers`, `--batch`, `--resume`.
  - `synthesis` — flags: `--objects` (file or comma list), `--num-objects` (alias `--n`), `--count`, `--rpm`, `--max-workers`, `--model`, `--api-base`, `--out`.
- `reader.py` — yields items from a folder or list file; items have `path` or `url` plus `id`.
- `vlm.py` — OpenAI-compatible client using `OpenAI`; loads `.env` via `dotenv.load_dotenv()`. Sends prompt + image (URL or JPEG data URL). Default temp 0.01, tokens 1024. Describe prompt returns NDJSON lines.
- `writer.py` — `JSONDirWriter` creates a unique run dir and writes one file per image; supports JSON via `write` and raw NDJSON via `write_text`. `JSONLWriter` kept for legacy.
- `utils/__init__.py` — `RateLimiter` (RPM, sliding window) for cross-thread throttling.
 - `detection.py` — `run_gdino_detection` (transformers pipeline) and `run_vlm_detection` (VLM JSON grounding) returning detections.
 - `synthesis.py` — `synthesize_one_line` generates one-line description plus per-object phrasings from a list of objects.

## Conventions/patterns
- Environment variables (auto-loaded):
  - `OBJH_API_KEY` (required), `OBJH_API_BASE` (optional), `OBJH_MODEL` (default `qwen/qwen2.5-vl-72b-instruct`), `OBJH_RPM` (0=unlimited)
- OpenAI client: `OpenAI(api_key=os.getenv("OBJH_API_KEY"), base_url=os.getenv("OBJH_API_BASE"))`
-- Image handling: open with Pillow, re-encode as JPEG, send either URL or `data:image/jpeg;base64,...`.
- Describe outputs per image: NDJSON lines, each line a single-key JSON object `{object: description}`.
- Detection outputs: `{ "image": str, "detections": [{ "label": str, "score": float, "bbox": {"xmin": float, "ymin": float, "xmax": float, "ymax": float} }] }` (pixel coordinates).
- Synthesis outputs: `{ "describe": str, "objects": [ {object: description}, ... ] }`.
- Concurrency: `ThreadPoolExecutor`; share one HTTP client; throttle with `RateLimiter` (–-rpm).
- Filenames: derived from basename or URL tail; sanitized; writer creates `out/run-YYYYMMDD-HHMMSS-<id>/`. With `--resume`, if `--out` points to a parent folder, the latest `run-*` folder is auto-selected; if `--out` is a specific `run-*` folder, it's used directly.

## Resume behavior
- `--resume` processes only items without existing outputs (NDJSON/JSON) in the target run dir.
- Existing outputs are detected by filename stem matches (same derivation used during write).

## Install & run
- Install CLI (recommended): `uv venv && source .venv/bin/activate && uv pip install -e .` → `object-harvest --help`
- Or run without install: `uv sync && uv run -m object_harvest.cli --help`
  - Try: `object-harvest describe --help`, `object-harvest detect --help`, `object-harvest synthesis --help`.

## Gotchas
- Don’t add duplicate log handlers; always use `get_logger`.
- Ensure `.env` is discoverable (CWD) or export vars in shell when running outside repo root.
- Set a reasonable `--rpm` with high `--max-workers` to avoid 429s; keep one `OpenAI` client per process.
- For `detect --backend gdino`, install `transformers` and `torch`, and set an appropriate HF model id.

References: `pyproject.toml`, `README.md`, `.env.example`, `src/object_harvest/*.py`.
