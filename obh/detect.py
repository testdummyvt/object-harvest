import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

from tqdm import tqdm

from obh.utils import setup_llm_client, rate_limited_call, encode_image_to_base64
from obh.utils.prompts import VLM_OBJECT_DET_SYS_PROMPT
from obh.utils.validation import validate_and_clean_vlm_response


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="Requests per minute limit (default: 60)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="LLM model to use (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for LLM API (default: OpenRouter)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (default: from OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object Harvest Detection")
    subparsers = parser.add_subparsers(dest="task", help="Available tasks")

    # VLM subcommand
    vlm_parser = subparsers.add_parser("vlm", help="Detect objects in images using VLM")
    vlm_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing images",
    )
    vlm_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    add_common_llm_args(vlm_parser)

    return parser.parse_args()


def vlm_task(args: argparse.Namespace) -> int:
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    if not image_paths:
        print(f"No image files found in {args.input}")
        return 1

    # Setup LLM client
    client = setup_llm_client(args.base_url, args.api_key)

    # Rate limiting setup
    rpm = args.rpm
    interval = 60 / rpm  # seconds between requests

    def process_image(img_path: str) -> Optional[Dict[str, Any]]:
        b64 = encode_image_to_base64(img_path)
        messages = [
            {"role": "system", "content": VLM_OBJECT_DET_SYS_PROMPT},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]},
        ]
        response = rate_limited_call(
            client,
            model=args.model,
            messages=messages,
            interval=interval,
        )
        try:
            data = validate_and_clean_vlm_response(response)
            return {"objects": data["objects"], "file_path": img_path}
        except ValueError as e:
            print(f"Error validating response for {img_path}: {e}")
            return None

    # Clear output file
    with open(args.output, "w") as f:
        pass

    batch_size = args.batch_size
    total_attempted = 0
    total_failed = 0
    total_successful = 0
    with ThreadPoolExecutor(max_workers=min(rpm, batch_size, os.cpu_count())) as executor:
        for start in range(0, len(image_paths), batch_size):
            batch_end = min(start + batch_size, len(image_paths))
            batch_paths = image_paths[start:batch_end]
            futures = [executor.submit(process_image, img_path) for img_path in batch_paths]
            batch_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {start//batch_size + 1}"):
                total_attempted += 1
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        total_successful += 1
                    else:
                        total_failed += 1
                except Exception as e:
                    print(f"Error in image processing: {e}")
                    total_failed += 1
            # Append batch results to file
            with open(args.output, "a") as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")

    print(f"Processed {total_successful} images to {args.output}")
    print("VLM detection summary:")
    print(f"  Total attempted: {total_attempted}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    return 0


def main() -> int:
    args = parse_args()
    if args.task == "vlm":
        return vlm_task(args)
    else:
        print("Error: No task specified. Use 'vlm'.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
