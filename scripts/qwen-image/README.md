# Qwen-Image helpers

Utilities and CLI scripts for generating images with the Qwen-Image diffusion model.

## What’s here

- `utils/qwen_image.py` — Thin wrapper around a Diffusers pipeline for Qwen-Image.
	- FlowMatch scheduler + fused Lightning LoRA weights for fast 8-step generation.
	- CUDA-only: moves pipeline to GPU and compiles the transformer for speed.
- `utils/prompt_enhance.py` — Prompt enhancement via the unified `AIClient`.
- `generate_images.py` — Generate an image per `describe` entry in synthesis NDJSON/JSONL.
- `convert_metadata_objects.py` — Migrate legacy `objects` formats to the new arrays schema.
- `upload_data.py` — Create & push a Hugging Face dataset (images + metadata) to the Hub.

## Setup

This repo uses `uv` for dependency management. To ensure the main project deps are installed:

```bash
uv sync
```

For Qwen-Image you’ll need a CUDA-capable environment and extra packages (torch, diffusers, xFormers, etc.). See `scripts/qwen-image/requirements.txt` for a reference list, or install suitable versions for your GPU/CUDA stack.

## Usage

### Generate images from synthesis NDJSON

`generate_images.py` reads a JSONL/NDJSON file where each line is a JSON object with a `describe` key (like the output from `src/object_harvest/synthesis.py`). It generates an image per line using Qwen-Image and saves them to a folder.

```bash
uv run python scripts/qwen-image/generate_images.py \
	--input /path/to/synthesis.jsonl \
	--out /tmp/qwen_images \
	--aspect-ratio 16:9 \
	--steps 8 \
	--randomize-seed \
	--format jpg
```

Options:

- `--input, -i` (required): Path to NDJSON from synthesis.
- `--out, -o` (required): Output folder for generated images.
- `--model-path`: HF model id or local path (default: `Qwen/Qwen-Image`).
- `--aspect-ratio`: Choose from the presets in `ASPECT_RATIO_SIZES` (e.g., `16:9`).
- `--steps`: Denoising steps (default: 8 with Lightning LoRA).
- `--seed`: Seed used when not randomizing.
- `--randomize-seed` / `--no-randomize-seed`: Enable/disable per-image random seed (default enabled).
- `--guidance-scale`: True CFG scale for the model (default: 1.0).
- `--format`: `jpg` or `png` (default: `jpg`).

Enhancement (optional):

- `--enhance`: Enable LLM-based prompt enhancement per `describe`.
- `--enhance-system-prompt`: System prompt to steer the enhancement (default: `SYS_PROMPT`).
- `--enhance-model`: Model to use for enhancement (default: uses `OBJH_MODEL`).
- `--enhance-base-url`: Custom API base URL (default: uses `OBJH_API_BASE`).
- `--enhance-temperature`: Enhancement temperature (default: 0.3).
- `--enhance-max-tokens`: Max tokens for enhancement output (default: 256).

Example with enhancement enabled:

```bash
uv run python scripts/qwen-image/generate_images.py \
	--input /path/to/synthesis.jsonl \
	--out /tmp/qwen_images \
	--enhance \
	--enhance-system-prompt "" \
	--aspect-ratio 16:9 \
	--steps 8 \
	--randomize-seed
```

The script shows a tqdm progress bar and logs saves using the project logger.

#### Output layout & metadata schema

Generation produces a directory like:

```
<out_dir>/
	data/                # image files (jpg/png)
	metadata.jsonl       # one JSON object per line
```

Each metadata line has fields:

| Field | Type | Description |
|-------|------|-------------|
| `describe` | string | Original description text from synthesis. |
| `objects.names` | array<string> | Ordered list of unique object names (first occurrence wins). |
| `objects.description` | array<string> | Parallel array of object descriptions (same indices as `names`). |
| `file_name` | string | Path relative to dataset root (e.g. `data/image-0001.jpg`). |
| `enhanced_describe` | string (optional) | LLM‑enhanced prompt actually used for generation (present only if enhancement changed text). |

Objects schema example:

```json
{
	"describe": "a cat sitting on a chair",
	"objects": {
		"names": ["cat", "chair"],
		"description": ["a small striped cat", "a wooden chair"]
	},
	"file_name": "data/coco-0001.jpg",
	"enhanced_describe": "a cozy scene of a striped cat perched on a wooden chair"
}
```

### Enhance a prompt (optional)

You can improve a text prompt before generation by calling the helper in code:

```python
from scripts.qwen-image.utils.prompt_enhance import enhance_prompt

enhanced = enhance_prompt(
		"a cozy cabin in the woods at dusk",
		system_prompt="",  # fill in guidance later if desired
)
```

This uses the unified `AIClient` and respects `OBJH_MODEL` / `OBJH_API_BASE` if a model/base URL isn’t passed.

### Converting legacy metadata (objects field)

Older runs may have one of these legacy `objects` encodings:

1. List of single-key dicts:
	```json
	{"objects": [{"cat": "a cat"}, {"dog": "a dog"}]} 
	```
2. Flat dict:
	```json
	{"objects": {"cat": "a cat", "dog": "a dog"}} 
	```
3. Nested dict wrapper:
	```json
	{"objects": {"objects": {"cat": "a cat", "dog": "a dog"}}}
	```

Current schema uses parallel arrays:

```json
{"objects": {"names": ["cat", "dog"], "description": ["a cat", "a dog"]}}
```

Run the converter (creates a `.bak` when in-place):

```bash
uv run python scripts/qwen-image/convert_metadata_objects.py --in-place path/to/metadata.jsonl
```

Or write side-by-side with suffix:

```bash
uv run python scripts/qwen-image/convert_metadata_objects.py -i metadata.jsonl --suffix converted
```

Multiple files:

```bash
uv run python scripts/qwen-image/convert_metadata_objects.py -i run1/metadata.jsonl run2/metadata.jsonl --in-place
```

### Uploading the dataset to Hugging Face Hub

After generation (and optional conversion), push the dataset:

```bash
uv run python scripts/qwen-image/upload_data.py \
  --dataset-dir /path/to/generated \
  --repo-id username/qwen-image-samples \
  --private
```

Requirements:
* `datasets` and `huggingface_hub` installed (`uv pip install datasets huggingface_hub`).
* HF token via `--token` or `HF_TOKEN` env var / cached login.

What it does:
1. Validates `metadata.jsonl` + `data/`.
2. Loads metadata into a Dataset.
3. Adds absolute image paths and casts to an Image column.
4. Pushes to the Hub (branch configurable via `--branch`).

### End-to-end workflow

1. Produce synthesis NDJSON (outside this directory) — lines with `describe` & `objects` (legacy forms accepted by converter if needed).
2. (Optional) Convert old metadata to arrays schema using `convert_metadata_objects.py`.
3. Generate images:
	```bash
	uv run python scripts/qwen-image/generate_images.py -i synthesis.jsonl -o ./generated --enhance --aspect-ratio 16:9
	```
4. (If synthesis used legacy objects lists) run converter on the produced `metadata.jsonl`.
5. Upload:
	```bash
	uv run python scripts/qwen-image/upload_data.py --dataset-dir ./generated --repo-id username/my-qwen-set
	```

Now the dataset is available on the Hub with structured objects arrays and image assets.

## Notes


## References

This code is mostly based on the following Hugging Face Spaces:

- multimodalart/Qwen-Image-Fast — https://huggingface.co/spaces/multimodalart/Qwen-Image-Fast
- multimodalart/Qwen-Image-LoRA-Explorer — https://huggingface.co/spaces/multimodalart/Qwen-Image-LoRA-Explorer
