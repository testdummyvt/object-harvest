# Object Detection to HuggingFace Dataset Converter

This script converts object detection JSON files to HuggingFace dataset format.

## Input Format

Each JSON file should contain:
```json
{
  "id": 1,
  "file_name": "image001.jpg",
  "detections": [
    {
      "label": "person",
      "score": 0.95,
      "bbox": {
        "xmin": 100.0,
        "ymin": 50.0,
        "xmax": 200.0,
        "ymax": 300.0
      }
    },
    {
      "label": "car", 
      "score": 0.88,
      "bbox": {
        "xmin": 300.0,
        "ymin": 200.0,
        "xmax": 500.0,
        "ymax": 400.0
      }
    }
  ]
}
```

## Output Format

The script converts to HuggingFace dataset format:
```json
{
  "id": 1,
  "file_name": "image001.jpg",
  "objects": {
    "label": ["person", "car"],
    "score": [0.95, 0.88],
    "bbox": [
      [100.0, 50.0, 200.0, 300.0],
      [300.0, 200.0, 500.0, 400.0]
    ]
  }
}
```

## Usage

### Basic Usage
```bash
python ovdet_to_hfdataset.py /path/to/json/folder /path/to/output/dataset
```

### Options
- `--format`: Output format (`arrow`, `json`, `parquet`). Default: `arrow`
- `--preview`: Preview first 3 samples without saving
- `--hf-repo-id`: Hugging Face repository ID to upload to (e.g., `username/dataset-name`)
- `--hf-token`: Hugging Face token for authentication (optional if logged in)
- `--hf-private`: Make the Hugging Face dataset private

### Examples

1. **Convert to Arrow format (default)**:
   ```bash
   python ovdet_to_hfdataset.py ./detections ./output_dataset
   ```

2. **Convert to JSON format**:
   ```bash
   python ovdet_to_hfdataset.py ./detections ./output_dataset.json --format json
   ```

3. **Preview without saving**:
   ```bash
   python ovdet_to_hfdataset.py ./detections ./output_dataset --preview
   ```

4. **Convert to Parquet format**:
   ```bash
   python ovdet_to_hfdataset.py ./detections ./output_dataset.parquet --format parquet
   ```

5. **Upload to Hugging Face Hub**:
   ```bash
   python ovdet_to_hfdataset.py ./detections ./output_dataset --hf-repo-id username/my-dataset
   ```

6. **Upload private dataset with token**:
   ```bash
   python ovdet_to_hfdataset.py ./detections ./output_dataset --hf-repo-id username/my-dataset --hf-token your_token --hf-private
   ```

## Requirements

Install the required dependencies:
```bash
pip install datasets
```

For Hugging Face Hub upload functionality:
```bash
pip install huggingface_hub
```

To authenticate with Hugging Face:
```bash
huggingface-cli login
```

Or provide your token directly with `--hf-token`.

## Testing

To test the conversion function with example data, uncomment the test line in the script:
```python
# Uncomment the line below to run the test
test_conversion()
```

Then run:
```bash
python ovdet_to_hfdataset.py
```