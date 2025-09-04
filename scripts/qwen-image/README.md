# Qwen-Image helpers

Utilities and CLI scripts for generating images with the Qwen-Image diffusion model.

## What’s here

- `utils/qwen_image.py` — A thin wrapper around a Diffusers pipeline for Qwen-Image.
	- Loads a FlowMatch scheduler and fuses Lightning LoRA weights for fast 8-step generation.
	- CUDA-only: moves pipeline to GPU and compiles the transformer for speed.
- `utils/prompt_enhance.py` — Small helper to enhance text prompts using the project’s `AIClient`.
- `generate_images.py` — CLI to read synthesis NDJSON/JSONL and generate an image per `describe` line.

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

## Notes


## References

This code is mostly based on the following Hugging Face Spaces:

- multimodalart/Qwen-Image-Fast — https://huggingface.co/spaces/multimodalart/Qwen-Image-Fast
- multimodalart/Qwen-Image-LoRA-Explorer — https://huggingface.co/spaces/multimodalart/Qwen-Image-LoRA-Explorer
