#!/usr/bin/env python3
"""
Convert object detection JSON files to Hugging Face dataset format.

Input JSON format:
{
    "id": int,
    "file_name": str,
    "detections": [
        {
            "label": str,
            "score": float,
            "bbox": {
                "xmin": float,
                "ymin": float,
                "xmax": float,
                "ymax": float
            }
        },
        ...
    ]
}

Output HuggingFace dataset format:
{
    "id": int,
    "file_name": str,
    "objects": {
        "label": [str, str, ...],
        "score": [float, float, ...],
        "bbox": [
            [xmin, ymin, xmax, ymax],
            [xmin, ymin, xmax, ymax],
            ...
        ]
    }
}
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset


def load_detection_json(file_path: Path) -> Dict[str, Any]:
    """Load a single object detection JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_detections_to_hf_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert detection format to HuggingFace dataset format.
    
    Args:
        data: Input data with 'id', 'file_name', and 'detections' keys
        
    Returns:
        Converted data with 'id', 'file_name', and 'objects' keys
    """
    detections = data.get('detections', [])
    
    # Extract lists for each object property
    labels = []
    scores = []
    bboxes = []
    
    for detection in detections:
        labels.append(detection['label'])
        scores.append(detection['score'])
        # Convert bbox dict to list [xmin, ymin, xmax, ymax]
        bbox = detection['bbox']
        bboxes.append([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
    
    return {
        'id': data['id'],
        'file_name': data['file_name'],
        'objects': {
            'label': labels,
            'score': scores,
            'bbox': bboxes
        }
    }


def process_json_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Process all JSON files in a folder and convert them to HF format.
    
    Args:
        folder_path: Path to folder containing JSON files
        
    Returns:
        List of converted data dictionaries
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    converted_data = []
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {folder_path}")
        return converted_data
    
    print(f"Processing {len(json_files)} JSON files...")
    
    for json_file in json_files:
        try:
            data = load_detection_json(json_file)
            converted = convert_detections_to_hf_format(data)
            converted_data.append(converted)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"Successfully processed {len(converted_data)} files")
    return converted_data


def create_huggingface_dataset(data_list: List[Dict[str, Any]]) -> Dataset:
    """
    Create a HuggingFace Dataset from the converted data.
    
    Args:
        data_list: List of converted data dictionaries
        
    Returns:
        HuggingFace Dataset object
    """
    if not data_list:
        raise ValueError("No data to create dataset from")
    
    # Prepare data for Dataset creation
    dataset_dict = {
        'id': [],
        'file_name': [],
        'objects': []
    }
    
    for item in data_list:
        dataset_dict['id'].append(item['id'])
        dataset_dict['file_name'].append(item['file_name'])
        dataset_dict['objects'].append(item['objects'])
    
    return Dataset.from_dict(dataset_dict)


def save_dataset(dataset: Dataset, output_path: str, format_type: str = "arrow") -> None:
    """
    Save the HuggingFace dataset to disk.
    
    Args:
        dataset: HuggingFace Dataset object
        output_path: Path where to save the dataset
        format_type: Format to save in ('arrow', 'json', 'parquet')
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == "arrow":
        dataset.save_to_disk(str(output_path_obj))
        print(f"Dataset saved to {output_path_obj} in Arrow format")
    elif format_type == "json":
        dataset.to_json(str(output_path_obj))
        print(f"Dataset saved to {output_path_obj} in JSON format")
    elif format_type == "parquet":
        dataset.to_parquet(str(output_path_obj))
        print(f"Dataset saved to {output_path_obj} in Parquet format")
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def upload_to_huggingface(dataset: Dataset, repo_id: str, token = None, private: bool = False) -> None:
    """
    Upload the dataset to Hugging Face Hub.
    
    Args:
        dataset: HuggingFace Dataset object
        repo_id: Repository ID on Hugging Face Hub (e.g., "username/dataset-name")
        token: Hugging Face token (if not provided, will try to use saved token)
        private: Whether to make the dataset private
    """
    try:
        print(f"🚀 Uploading dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private
        )
        print(f"✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload to Hugging Face Hub: {e}")
        print("💡 Make sure you have:")
        print("   1. Installed huggingface_hub: pip install huggingface_hub")
        print("   2. Logged in: huggingface-cli login")
        print("   3. Or provided a valid token with --hf-token")
        raise


def main():
    """Main function to handle command line arguments and orchestrate the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert object detection JSON files to HuggingFace dataset format"
    )
    parser.add_argument(
        "input_folder",
        help="Path to folder containing object detection JSON files"
    )
    parser.add_argument(
        "output_path",
        help="Path where to save the HuggingFace dataset"
    )
    parser.add_argument(
        "--format",
        choices=["arrow", "json", "parquet"],
        default="arrow",
        help="Output format for the dataset (default: arrow)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the first few converted samples without saving"
    )
    parser.add_argument(
        "--hf-repo-id",
        help="Hugging Face repository ID to upload to (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for authentication (optional if logged in via CLI)"
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Make the Hugging Face dataset private"
    )
    
    args = parser.parse_args()
    
    try:
        # Process JSON files
        converted_data = process_json_folder(args.input_folder)
        
        if not converted_data:
            print("No data to process. Exiting.")
            return
        
        # Create HuggingFace dataset
        dataset = create_huggingface_dataset(converted_data)
        
        print(f"\nDataset created with {len(dataset)} samples")
        print(f"Dataset columns: {dataset.column_names}")
        print(f"Dataset features: {dataset.features}")
        
        # Preview mode
        if args.preview:
            print("\nPreview of first 3 samples:")
            preview_samples = dataset.select(range(min(3, len(dataset))))
            for i in range(len(preview_samples)):
                sample = preview_samples[i]
                print(f"\nSample {i + 1}:")
                print(f"  ID: {sample['id']}")
                print(f"  File: {sample['file_name']}")
                print(f"  Objects: {len(sample['objects']['label'])} detections")
                if sample['objects']['label']:
                    print(f"  First detection: {sample['objects']['label'][0]} "
                          f"(score: {sample['objects']['score'][0]:.3f})")
            return
        
        # Save dataset
        save_dataset(dataset, args.output_path, args.format)
        
        print(f"\n✅ Successfully converted {len(converted_data)} JSON files to HuggingFace dataset!")
        
        # Upload to Hugging Face Hub if requested
        if args.hf_repo_id:
            upload_to_huggingface(
                dataset=dataset, 
                repo_id=args.hf_repo_id,
                token=args.hf_token,
                private=args.hf_private
            )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def create_example_data() -> Dict[str, Any]:
    """Create example data for testing purposes."""
    return {
        "id": 1,
        "file_name": "image001.jpg",
        "detections": [
            {
                "label": "person",
                "score": 0.95,
                "bbox": {"xmin": 100.0, "ymin": 50.0, "xmax": 200.0, "ymax": 300.0}
            },
            {
                "label": "car",
                "score": 0.88,
                "bbox": {"xmin": 300.0, "ymin": 200.0, "xmax": 500.0, "ymax": 400.0}
            }
        ]
    }


def test_conversion():
    """Test the conversion function with example data."""
    example_data = create_example_data()
    converted = convert_detections_to_hf_format(example_data)
    
    print("Original format:")
    print(json.dumps(example_data, indent=2))
    
    print("\nConverted to HuggingFace format:")
    print(json.dumps(converted, indent=2))
    
    # Verify the conversion
    assert converted['id'] == example_data['id']
    assert converted['file_name'] == example_data['file_name']
    assert len(converted['objects']['label']) == len(example_data['detections'])
    assert converted['objects']['label'] == ['person', 'car']
    assert converted['objects']['score'] == [0.95, 0.88]
    assert converted['objects']['bbox'] == [[100.0, 50.0, 200.0, 300.0], [300.0, 200.0, 500.0, 400.0]]
    
    print("\n✅ Test passed!")


if __name__ == "__main__":
    # Uncomment the line below to run the test
    # test_conversion()
    main()