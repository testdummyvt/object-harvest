# Object Harvest

Object Harvest is a tool for generating and processing object-related data, including prompt generation using LLMs, and object detection in images.

## Features

- **Prompt Generation (`prompt-gen`)**: Generate scene descriptions containing specified objects using OpenAI-compatible LLMs (e.g., via OpenRouter).
- **VLM Object Detection (`vlm`)**: Detect objects in images using Vision-Language Models (VLMs) with bounding boxes.
- **Image Captioning (`moondream-caption`)**: Generate detailed image captions using the Moondream API (Cloud or Local).
- **Multi-threading**: Efficient parallel processing with configurable rate limiting.
- **NDJSON Output**: Structured output for easy processing.

## Installation

1. Ensure you have Python ≥ 3.12 installed.
2. Clone the repository and navigate to the project directory.
3. Install dependencies using `uv`:

   ```bash
   uv venv
   uv pip install -e ."[all]"
   ```

## Setup

### API Keys
For LLM access, set your API key (typically from OpenRouter) as an environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

For Moondream image captioning (cloud mode), set:

```bash
export MOONDREAM_API_KEY="your-api-key-here"
```

Or pass them via the `--api-key` CLI argument for the respective commands.

## Usage

### General Command Structure

For generation tasks:
```bash
uv run python -m obh.generate <task> [options]
```

Or using the installed script:
```bash
obh-generate <task> [options]
```

For detection tasks:
```bash
uv run python -m obh.detect <task> [options]
```

Or using the installed script:
```bash
obh-detect <task> [options]
```

### Tasks

#### Prompt Generation (`prompt-gen`)

Generates scene descriptions containing the specified objects using an LLM.

**Required Arguments:**
- `--num-prompts`: Number of prompts to generate
- `--output`: Path to output NDJSON file

**Object Input (choose one):**
- `--objects-file`: Text file with objects (one per line, format: `object — descriptor`)
- `--objects-list`: Comma-separated list (format: `object — descriptor`)

**Optional Arguments:**
- `--rpm`: Requests per minute limit (default: 60)
- `--model`: LLM model (default: `openai/gpt-4o`)
- `--base-url`: API base URL (default: `https://openrouter.ai/api/v1`)
- `--api-key`: API key (overrides env var)
- `--batch-size`: Batch size for processing (default: 10)
- `--min-objects`: Minimum number of objects to randomly select per prompt (must be used with `--max-objects`)
- `--max-objects`: Maximum number of objects to randomly select per prompt (must be used with `--min-objects`)

**Example:**

```bash
# Using objects list with random object selection
uv run python -m obh.generate prompt-gen \
  --objects-list "apple — red, shiny; banana — yellow, curved; orange — orange, round" \
  --num-prompts 5 \
  --min-objects 1 \
  --max-objects 2 \
  --output prompts.ndjson

# Using objects file with fixed batch size
echo "apple — red, shiny" > objects.txt
echo "banana — yellow, curved" >> objects.txt
echo "orange — orange, round" >> objects.txt
uv run python -m obh.generate prompt-gen \
  --objects-file objects.txt \
  --num-prompts 10 \
  --rpm 30 \
  --batch-size 10 \
  --output prompts.ndjson
```

**Output Format:**
Each line in the NDJSON file is a JSON object with:
- `describe`: The generated scene description
- `objects`: Array of objects with their exact phrasing in the description

**Processing Details:**
- Uses multi-threaded batch processing for efficient generation
- Displays progress bars for each batch
- Prints a summary of attempted, successful, and failed generations at the end

#### Image Captioning (`moondream-caption`)

Generates image captions using the Moondream API, supporting both cloud (via API key) and local servers.

**Required Arguments:**
- `--input`: Path to input directory containing images or a single image file
- `--output`: Path to output NDJSON file

**Optional Arguments:**
- `--length`: Caption length: `short`, `normal`, or `long` (default: `normal`)
- `--local`: Use local Moondream server at `http://localhost:2020/v1` (default: False)
- `--api-key`: Moondream API key (overrides env var)
- `--rpm`: Requests per minute limit (default: 60)
- `--batch-size`: Batch size for processing (default: 10)
- `--sequential`: Process images sequentially (useful for local models where parallel processing might overwhelm resources)

**Example:**

```bash
# Caption images using Moondream Cloud
uv run python -m obh.generate moondream-caption \
  --input images/ \
  --output captions.ndjson \
  --length long \
  --rpm 60

# Caption a single image using a local Moondream server
uv run python -m obh.generate moondream-caption \
  --input sample.jpg \
  --output caption.ndjson \
  --local
```

**Output Format:**
Each line in the NDJSON file is a JSON object with:
- `file_path`: Path to the processed image file
- `caption`: The generated caption text

**Processing Details:**
- Processes images recursively from the input directory
- Supports `.jpg`, `.jpeg`, `.png`, and `.webp` formats
- Uses multi-threaded batch processing with configurable rate limiting
- Displays progress bar and summary statistics


### Detection Tasks

#### VLM Object Detection (`vlm`)

Detects objects in images using Vision-Language Models (VLMs) and outputs bounding boxes for each detected object.

**Required Arguments:**
- `--input`: Directory containing images (supports .jpg, .jpeg, .png)
- `--output`: Path to output JSONL file

**Optional Arguments:**
- `--rpm`: Requests per minute limit (default: 60)
- `--model`: VLM model (default: `openai/gpt-4o`)
- `--base-url`: API base URL (default: `https://openrouter.ai/api/v1`)
- `--api-key`: API key (overrides env var)
- `--batch-size`: Batch size for processing (default: 10)

**Example:**

```bash
# Detect objects in images using VLM
uv run python -m obh.detect vlm \
  --input images/ \
  --output detections.jsonl \
  --rpm 30 \
  --batch-size 10
```

**Output Format:**
Each line in the JSONL file is a JSON object with:
- `objects`: Object with `labels` (array of object names) and `bbox` (array of bounding boxes as [x_min, y_min, x_max, y_max])
- `file_path`: Path to the processed image file

**Processing Details:**
- Recursively finds image files in the input directory
- Uses multi-threaded batch processing for efficient detection
- Displays progress bars for each batch
- Validates responses and handles errors gracefully
- Prints a summary of attempted, successful, and failed detections at the end

## Development

### Code Style
- Use `uv run ruff check --fix .` to lint and format code.
- Run `uv run pytest` for tests (add tests under `tests/`).

### Project Structure
- `obh/`: Main package
  - `generate.py`: CLI entry point for generation tasks
  - `detect.py`: CLI entry point for detection tasks
  - `utils/`: Shared utilities
    - `llm_utils.py`: LLM-related helper functions
    - `moondream_utils.py`: Moondream API helper functions
    - `validation.py`: Response validation functions
- `tests/`: Unit tests

## Contributing

Follow the guidelines in `AGENTS.md` for code style, testing, and PR workflow.
