# Object Harvest

Extract lists of objects and optionally bounding boxes from images using Vision Language Models (VLMs) using openai-api and OpenRouter. Supports local folders and list files(text). Outputs one JSON line per image with structured data and metadata.

## Features

- Supports input sources from folder, text list files.
- OpenAI-compatible API client (supply `--api-base` for alternate providers)
- Multithreaded processing for faster throughput
