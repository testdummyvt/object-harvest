# AI agent guide for object-harvest

Purpose: Extract NDJSON object suggestions from images using Vision-Language Models (VLMs), perform open-vocabulary detection (HF zero-shot models like GroundingDINO/LLMDet) over Hugging Face datasets (sequential, dataset-only), and synthesize one-line descriptions—writing one file per image.

## Snapshot
- Python 3.12, src-layout: `src/object_harvest/`
- CLI present: `object-harvest` (entry → `object_harvest.cli:main`)
- Deps: `openai`, `pillow`, `tqdm`, `python-dotenv` (see `pyproject.toml`); `uv.lock` for env sync
- Config via `.env` (auto-loaded): `OBJH_API_KEY`, `OBJH_API_BASE`, `OBJH_MODEL`, `OBJH_RPM`

### Environment Activation (Required)
Always ensure the project's virtual environment is active before running any CLI commands, tests, or Python scripts. Activate with:

```bash
source .venv/bin/activate
```

If the environment does not exist yet:

```bash
uv venv
source .venv/bin/activate
uv sync
```

All subsequent examples assume the environment is active.

## Modules and roles
- `logging.py` — emoji logger. Use `get_logger(__name__)` once per module (idempotent handlers).
- `cli.py` — subcommands:
  - `describe` — emits per-image `.ndjson` object suggestions.
  - `ovdet` — dataset-only open-vocabulary detection (requires `--hf-dataset`), sequential (no thread pool). Flags: `--input` (placeholder), `--out`, `--hf-model`, `--hf-dataset`, `--hf-dataset-split`, `--use-obj-desp`, `--objects` (ignored in dataset mode), `--threshold`, `--resume`.
  - `synthesis` — generates one-line descriptions from object lists.
- `reader.py` — yields items from a folder or list file; items have `path` or `url` plus `id`.
- `vlm.py` — VLM prompt logic using the unified `AIClient` (from `utils/clients.py`); loads `.env` via `dotenv.load_dotenv()`. Sends prompt + image (URL or JPEG data URL). Default temp 0.01, tokens 1024. Describe prompt returns NDJSON lines.
- `writer.py` — `JSONDirWriter` creates a unique run dir and writes one file per image; supports JSON via `write` and raw NDJSON via `write_text`. `JSONLWriter` kept for legacy.
- `utils/__init__.py` — `RateLimiter` (RPM, sliding window), JSONL batch helpers, tqdm GPM helper.
- `utils/clients.py` — `AIClient` unified OpenAI-compatible client to be shared across threads.
- `detection.py` — classes `HFDataLoader` (iterates HF dataset, builds prompts) and `OVDModel` (wraps processor/model, auto device selection, sequential inference).
- `synthesis.py` — `synthesize_one_line` generates one-line description plus per-object phrasings from a list of objects; accepts an optional `AIClient` for reuse.

## Conventions/patterns
- Environment variables (auto-loaded):
  - `OBJH_API_KEY` (required), `OBJH_API_BASE` (optional), `OBJH_MODEL` (default `qwen/qwen2.5-vl-72b-instruct`), `OBJH_RPM` (0=unlimited)
- Unified client: `AIClient(model, base_url)` wraps `OpenAI(api_key=os.getenv("OBJH_API_KEY"), base_url=os.getenv("OBJH_API_BASE"))`; reuse a single instance per process.
-- Image handling: open with Pillow, re-encode as JPEG, send either URL or `data:image/jpeg;base64,...`.
- Describe outputs per image: NDJSON lines, each line a single-key JSON object `{object: description}`.
- OVDet inputs: requires Hugging Face dataset (`--hf-dataset`, plus optional `--hf-dataset-split`).
- OVDet outputs: `{ "id": int, "file_name": str, "detections": [{ "label": str, "score": float, "bbox": {"xmin": float, "ymin": float, "xmax": float, "ymax": float} }] }`.
- Synthesis outputs: `{ "describe": str, "objects": [ {object: description}, ... ] }`.
- Concurrency: ovdet sequential (no thread pool) for deterministic ordering and simpler memory profile. Describe & synthesis still use threads. Synthesis shows GPM; can batch-append JSONL with `--save-batch-size`.
- Filenames: derived from basename or URL tail; sanitized; writer creates `out/run-YYYYMMDD-HHMMSS-<id>/`. With `--resume`, if `--out` points to a parent folder, the latest `run-*` folder is auto-selected; if `--out` is a specific `run-*` folder, it's used directly.

## Resume behavior
- `--resume` processes only items without existing outputs (NDJSON/JSON) in the target run dir.
- Existing outputs are detected by filename stem matches (same derivation used during write).

## Install & run
- Install CLI (recommended): `uv venv && source .venv/bin/activate && uv pip install -e .` → `object-harvest --help`
- Or run without install: `uv sync && uv run -m object_harvest.cli --help`
  - Try: `object-harvest describe --help`, `object-harvest ovdet --help`, `object-harvest synthesis --help`.

## Gotchas
- Don’t add duplicate log handlers; always use `get_logger`.
- Ensure `.env` is discoverable (CWD) or export vars in shell when running outside repo root.
- Set a reasonable `--rpm` with high `--max-workers` to avoid 429s; keep one `OpenAI` client per process.
- For detection, install `transformers` and `torch`, and set an appropriate HF model id (e.g., `iSEE-Laboratory/llmdet_large`). A Hugging Face dataset id is required.

References: `pyproject.toml`, `README.md`, `.env.example`, `src/object_harvest/*.py`.
