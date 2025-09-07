## Cleaning malformed `describe` fields in synthesis outputs

Large VLMs sometimes return a `describe` value that accidentally embeds an entire JSON snippet instead of a plain sentence. For example, instead of a string, the model may echo something like:

```json
{
	"describe": "{\"describe\": \"a cat on a red couch\", \"objects\": [{\"cat\": \"tabby cat\"}]}",
	"objects": [{"cat": "tabby cat"}, {"couch": "red couch"}]
}
```

This breaks downstream consumers that expect `describe` to be just a sentence. We provide a cleaner script to fix these cases by extracting the inner description.

### How the cleaning works

The logic looks for evidence of nested JSON (the substrings `"describe"` and/or `"objects"`) inside the `describe` string and then tries, in order:

1. Parse the entire string as JSON and read the inner `describe` key.
2. Use a regex to capture the value of the inner `"describe": "..."` and JSON-decode it to handle escapes.
3. If `"objects"` is present, take the substring before it and peel quotes/braces as a fallback.
4. If none of the above works, keep the original string.

This approach is robust to minor formatting differences while preserving valid strings unchanged.

### Script: `scripts/clean_describes/qwen3_235B.py`

The repository includes a ready-to-use cleaner with logging and a tqdm progress bar:

- Reads an input JSONL file.
- For each JSON object, if `describe` contains `"describe"` or `"objects"`, it extracts the inner description.
- Writes cleaned JSONL to stdout or an output path.
- Reports totals, modified lines, and malformed JSON lines.

Usage (examples):

```bash
python scripts/clean_describes/qwen3_235B.py -i data/synth.jsonl -o data/synth.cleaned.jsonl
# or write to stdout
python scripts/clean_describes/qwen3_235B.py -i data/synth.jsonl
```

Notes:
- The same cleaning strategy applies to outputs from other models (e.g., Qwen3 235B). If you maintain a model-specific script (e.g., `qwen3_235B.py`), mirror the same `extract_inner_describe` and JSONL iteration pattern used here.
- The cleaner is conservative: it only rewrites when it finds clear signs of nested JSON; otherwise it keeps the original `describe`.

### Before/After example

Input line:

```json
{"describe": "{\"describe\": \"a cat on a red couch\", \"objects\": [{\"cat\": \"tabby\"}]}", "objects": [{"cat": "tabby"}, {"couch": "red"}]}
```

Cleaned line:

```json
{"describe": "a cat on a red couch", "objects": [{"cat": "tabby"}, {"couch": "red"}]}
```

