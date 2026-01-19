import click
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

from tqdm import tqdm

from obh.utils import setup_llm_client, rate_limited_call, encode_image_to_base64
from obh.utils.prompts import VLM_OBJECT_DET_SYS_PROMPT
from obh.utils.validation import validate_and_clean_vlm_response


# Common Click options for LLM configuration
def add_common_llm_options(func):
    """Decorator to add common LLM options to Click commands."""
    func = click.option(
        "--rpm",
        type=int,
        default=60,
        help="Requests per minute limit (default: 60)",
    )(func)
    func = click.option(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="LLM model to use (default: openai/gpt-4o)",
    )(func)
    func = click.option(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for LLM API (default: OpenRouter)",
    )(func)
    func = click.option(
        "--api-key",
        type=str,
        help="API key (default: from OPENROUTER_API_KEY env var)",
    )(func)
    func = click.option(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )(func)
    return func


@click.group()
def cli():
    """Object Harvest Detection CLI."""
    pass


@cli.command()
@click.option(
    "--input",
    type=str,
    required=True,
    help="Input directory containing images",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Output JSONL file path",
)
@add_common_llm_options
def vlm(input: str, output: str, rpm: int, model: str, base_url: str, api_key: str, batch_size: int) -> None:
    """Detect objects in images using VLM."""
    # Create args namespace to maintain compatibility with existing vlm_task function
    from argparse import Namespace
    args = Namespace(
        input=input,
        output=output,
        rpm=rpm,
        model=model,
        base_url=base_url,
        api_key=api_key,
        batch_size=batch_size
    )
    result = vlm_task(args)
    if result != 0:
        raise click.ClickException(f"VLM task failed with exit code {result}")


def vlm_task(args) -> int:
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
            labels = [obj["labels"] for obj in data["objects"]]
            bboxes = [obj["bbox_2d"] for obj in data["objects"]]
            return {"objects": {"labels": labels, "bbox": bboxes}, "file_path": img_path}
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
    # Ensure rpm and batch_size are not None before using min
    cpu_count = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    max_workers = min(rpm, batch_size, cpu_count)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    # Use Click's command invocation
    # The cli() function will handle command routing
    cli(prog_name='obh-detect')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
