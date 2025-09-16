# Object Harvest

Extract object suggestions from images using Vision-Language Models (VLMs), perform open-vocabulary detection, and synthesize descriptions. Inputs can be a folder of images or a text file of paths/URLs. Describe now writes one NDJSON file per image (one JSON object per line), saved under a unique run folder.

## Features

- Inputs: folder of images or text list file (paths and/or URLs)
- Outputs: one file per image under a unique run directory (run-YYYYMMDD-HHMMSS-<id>)
  - Describe: .ndjson (one line per object: {"object": "short description"})
  - OVDet: .json (detections array with XYXY pixel bboxes)
- OpenAI-compatible API client (set `--api-base` for OpenRouter/others)
- Default model: `qwen/qwen2.5-vl-72b-instruct` (override via `--model` or `OBJH_MODEL`)
- Concurrency via threads + a shared RPM rate limiter (`--rpm`)
- Progress bar with live Generations Per Minute (GPM) for synthesis
- Batched JSONL saving for synthesis via `--save-batch-size` to improve durability with many workers
- Subcommands:
  - `describe` — suggest objects as NDJSON lines (includes people when present)
  - `ovdet` — open-vocabulary detection via Hugging Face models (GroundingDINO/LLMDet) over HF datasets (sequential)
  - `synthesis` — generate one-line descriptions from a list of objects

### Qwen-Image helpers (scripts)

This repo also includes optional helpers for image generation and dataset publishing under `scripts/qwen-image/`:

- `generate_images.py` — read synthesis NDJSON/JSONL (`describe` + legacy objects) and generate one image per line using Qwen-Image. Produces `metadata.jsonl` + `data/` images. Objects are stored in the new arrays schema: `{ "objects": { "names": [...], "description": [...] } }`.
- `convert_metadata_objects.py` — migrate legacy `objects` encodings (list of dicts, flat dict, nested) into the arrays schema.
- `upload_data.py` — create a Hugging Face `datasets` dataset from `metadata.jsonl` + images and push it to the Hub.

See full usage in `scripts/qwen-image/README.md`.

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
object-harvest ovdet --help
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
```

### OVDet (open-vocabulary detection)

Current implementation (latest changes) supports only Hugging Face dataset inputs. The old image/JSONL direct modes and free-form text prompting were removed in favor of a simpler, deterministic dataset-driven pipeline. Detection now runs sequentially (no thread pool) and shows a progress bar over the dataset.

Minimal example:

```bash
object-harvest ovdet \
  --input unused_but_required \
  --hf-dataset your-namespace/your_dataset_id \
  --hf-dataset-split train \
  --out ./detections \
  --hf-model iSEE-Laboratory/llmdet_large \
  --threshold 0.35
```

Key flags (ovdet):

- `--hf-dataset <id>`: Required. Hugging Face dataset repository id.
- `--hf-dataset-split <split>`: Dataset split to load (default `train`).
- `--hf-model <model_id>`: HF zero-shot OD model (e.g. `iSEE-Laboratory/llmdet_large`).
- `--use-obj-desp`: Use `objects.description` instead of `objects.names` as prompts.
- `--objects <file|comma,list>`: (Present but currently ignored in dataset mode; prompts are taken from dataset objects.)
- `--threshold <float>`: Score threshold (applied during post-processing).
- `--resume`: Skip writing detections for items already present in the output run directory.

Dataset object schema requirements:

```jsonc
{
  "file_name": "relative/or/original/path/or/name.jpg", // optional if image object contains path metadata
  "image": <PIL Image or dataset image feature>,
  "objects": {
    "names": ["cat", "sofa", "lamp"],
    "description": ["black cat lounging", "fabric sofa", "tall floor lamp"]
  }
}
```

Prompt source selection:
- By default prompts come from `objects.names`.
- If `--use-obj-desp` is set, prompts come from `objects.description`.

Execution model:
- Sequential processing; each sample is loaded, prompts extracted, fed to the model, detections written immediately.
- Device auto-selection: CUDA → MPS → CPU.
- A one-time log line reports the model id and selected device.

Outputs (per image record):

```json
{
  "id": 42,
  "file_name": "example.jpg",
  "detections": [
    {"label": "cat", "score": 0.91, "bbox": {"xmin": 10.0, "ymin": 20.0, "xmax": 120.0, "ymax": 240.0}}
  ]
}
```

Note: Legacy flags (`--from-describe`, `--text`, image/JSONL direct inputs, concurrency settings) are no longer active in the current OVDet implementation.

### Synthesis (generate one-line description + per-object phrasings)

Generate a one-line description that includes N objects from a provided list, and also return a per-object description phrased the same way as in the main description:

```bash
object-harvest synthesis \
  --objects ./objects.txt \
  --num-objects 6 \
  --count 10 \
  --rpm 60 \
  --max-workers 8 \
  --model qwen/qwen2.5-vl-72b-instruct \
  --out ./synthesis.jsonl \
  --save-batch-size 25
```

Flags (synthesis):
- `--objects <file|comma,list>` unified input (file path or comma-separated list)
- `--num-objects` (alias `--n`) number of objects to include per description (random sampled)
- `--count` number of samples to generate; with `--out` ending in .jsonl, writes one JSON object per line; otherwise writes a JSON array/file
- `--rpm`, `--max-workers` for throughput control
- `--out` optional; if omitted, prints to stdout
 - `--save-batch-size <N>` (only for .jsonl) append results in batches of N for durability during long runs

Note:
- The synthesis prompt template was updated to encourage short visual descriptors and strict JSON structure from the LLM. The CLI still returns legacy per-object entries as a list of single-key dicts for compatibility with existing consumers. The Qwen-Image generation pipeline normalizes these into arrays.


## Output

The CLI writes a unique run directory under `--out`, e.g. `out/run-20250822-104455-ab12cd34/`.

Describe (.ndjson per image):
```json
{"bicycle": "a blue road bike leaning against a brick wall"}
{"person": "a man wearing a red jacket walking beside the bike"}
{"backpack": "black backpack with a water bottle in side pocket"}
```

OVDet (.json per image):
```json
{
  "image": "file_or_url",
  "detections": [
    {"label": "person", "score": 0.91, "bbox": {"xmin": 10.0, "ymin": 20.0, "xmax": 120.0, "ymax": 240.0}},
    {"label": "bicycle", "score": 0.88, "bbox": {"xmin": 40.0, "ymin": 60.0, "xmax": 300.0, "ymax": 220.0}}
  ]
}
```

Synthesis (.json or .jsonl records):
```json
{
  "describe": "A black cat lounges on a soft fabric sofa by the window.",
  "objects": [
    {"cat": "black cat lounging"},
    {"sofa": "soft fabric sofa"}
  ]
}
```

When writing `.jsonl` with `--save-batch-size`, results are appended in batches for resilience. A live progress bar shows GPM.

### Generated image dataset format (Qwen-Image helpers)

When using `scripts/qwen-image/generate_images.py`, outputs are organized as:

```
<out_dir>/
  data/                # image files
  metadata.jsonl       # one JSON object per image
```

Each metadata line contains at least:

```json
{
  "describe": "a cat sitting on a chair",
  "objects": {
    "names": ["cat", "chair"],
    "description": ["a small striped cat", "a wooden chair"]
  },
  "file_name": "data/coco-0001.jpg"
}
```

- If prompt enhancement is enabled, an additional `enhanced_describe` field is included.
- The `convert_metadata_objects.py` script can migrate earlier `objects` formats to this arrays schema.

## Notes

- Describe is production-ready; OVDet supports HF zero-shot models (e.g., LLMDet/GroundingDINO). Synthesis uses the LLM API.
- A single OpenAI-compatible client is shared across threads. Use `--rpm` to avoid provider rate limits.
- Supported image types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`.

### Best practices
- Reuse one client per process to avoid exhausting file descriptors/sockets (fixes errors like `[Errno 24] Too many open files`). The CLI already does this for synthesis.
- With high `--max-workers`, set a non-zero `--rpm` to reduce 429s.

## Development

- Code lives in `src/object_harvest/`.
  - `cli.py`: argument parsing and orchestration
  - `reader.py`: iterates inputs (folder or list file)
  - `vlm.py`: prompt logic for NDJSON describe using the unified client
  - `detection.py`: HF zero-shot detection (GroundingDINO/LLMDet)
  - `synthesis.py`: one-line description + per-object phrasing generation
  - Prompt template enforces strict JSON and descriptor usage; return format remains a list of single-key dicts for compatibility.
  - `utils/clients.py`: unified OpenAI-compatible client (`AIClient`)
  - `utils/__init__.py`: `RateLimiter`, JSONL batch helpers, tqdm GPM helper
  - `writer.py`: writes per-image JSON files; also supports raw NDJSON via `write_text`
  - `logging.py`: emoji-prefixed logger (use `get_logger(__name__)`)

## Troubleshooting

- 401/403 errors: verify `OBJH_API_KEY` and `OBJH_API_BASE`.
- 429 errors: reduce `--max-workers` and/or set `--rpm`.
- Detection backends: ensure `transformers` and `torch` are installed and the chosen HF model is available. OVDet requires a Hugging Face dataset (`--hf-dataset`).
- Module not found: ensure `uv run -m object_harvest.cli` or `PYTHONPATH=src` when using `python -m ...`.
 