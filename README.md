# object-harvest

Extract lists of objects (and optionally bounding boxes) from images using OpenAI-compatible
Vision Language Models (VLMs). Supports local folders, list files, and HuggingFace datasets.
Outputs one JSON line per image with structured data and metadata.

## Features
- Multiple ingestion sources: directory, text list file, HuggingFace dataset (optional `datasets` dep)
- OpenAI-compatible API client (supply `--api-base` for alternate providers)
- Two-phase or single-phase prompting: object list then bounding boxes
- Multithreaded processing for faster throughput
- Robust JSON parsing with markdown fence & partial repair handling
- Emoji-enhanced logging (🟢, ⚠️, ❌, 📊, 🚀)
- Deterministic mockable pipeline for tests

## Quick Start (uv)
```bash
uv venv
source .venv/bin/activate
uv pip install -e .[openai]
```

## CLI Usage
```bash
object-harvest --source ./images --model gpt-4o-mini --output harvest.jsonl --boxes \
		--threads 6 --api-key-env OPENAI_API_KEY
```

Alternate ingestion examples:
```bash
# From list file
object-harvest --list-file images.txt --model gpt-4o-mini --output out.jsonl

# HuggingFace dataset (needs datasets installed)
object-harvest --dataset cifar10 --dataset-split test --model gpt-4o-mini --max-images 100
```

Important flags:
- `--source DIR` directory of images
- `--list-file FILE` newline-separated image paths
- `--dataset NAME` HuggingFace dataset (image column auto-detected)
- `--dataset-split SPLIT` dataset split (default `train`)
- `--model MODEL` OpenAI-compatible model name
- `--boxes` run second pass for bounding boxes
- `--threads N` threads for parallel API calls (default 4)
- `--api-base URL` override base URL (e.g. for OpenRouter)
- `--api-key-env ENVVAR` environment variable containing API key (default `OPENAI_API_KEY`)
- `--max-images N` limit for sampling

Output JSONL record example:
```json
{
	"image_id": "dog-1",
	"path": "images/dog.jpg",
	"model": "gpt-4o-mini",
	"objects": [{"name": "dog", "confidence": 0.92}],
		"boxes": [{"name": "dog", "x1": 34, "y1": 51, "x2": 182, "y2": 256}],
	"t_total": 1.234,
	"attempts": 1
}
```

## Development
Install all extras and dev tools:
```bash
uv pip install -e .[all]
```

Run tests:
```bash
pytest -q
```

Format & lint (if ruff/black installed):
```bash
ruff check .
ruff format .  # or black .
```

## Mock / Testing Notes
Tests monkeypatch the `VLMClient` to avoid network calls. Provide a real API key via env var for live runs.

## Roadmap
- Rate limiting & retry policies
- Video frame extraction
- Unified objects+boxes single-call prompt option
- Enhanced bounding box validation & visualization helpers

---
See `.github/copilot-instructions.md` for architecture & contribution patterns.
