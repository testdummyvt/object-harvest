import os
from typing import Optional, Dict, Any
import random
import click
import glob
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from obh.utils import (
    load_objects,
    setup_llm_client,
    rate_limited_call,
    setup_moondream_client,
    rate_limited_caption,
    encode_image_to_base64,
)
from obh.utils.prompts import PROMPTGEN_SYS_PROMPT, CAPTION_SYS_PROMPT
from obh.utils.validation import (
    validate_and_clean_prompt_gen_response,
    restructure_objects,
)


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
    """Object Harvest Generator CLI."""
    pass


@cli.command()
@click.option(
    "--objects-file",
    type=str,
    help="Path to text file with objects (one per line, format: 'object — descriptor')",
)
@click.option(
    "--objects-list",
    type=str,
    help="Comma-separated list of objects (format: 'object — descriptor')",
)
@click.option(
    "--num-prompts",
    type=int,
    required=True,
    help="Number of prompts to generate",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Output NDJSON file path",
)
@click.option(
    "--min-objects",
    type=int,
    help="Minimum number of objects to randomly select per prompt (optional)",
)
@click.option(
    "--max-objects",
    type=int,
    help="Maximum number of objects to randomly select per prompt (optional)",
)
@add_common_llm_options
def prompt_gen(
    objects_file: str,
    objects_list: str,
    num_prompts: int,
    output: str,
    min_objects: int,
    max_objects: int,
    rpm: int,
    model: str,
    base_url: str,
    api_key: str,
    batch_size: int,
) -> None:
    """Generate prompts using LLM."""
    # Create args namespace to maintain compatibility with existing prompt_gen_task function
    from argparse import Namespace

    args = Namespace(
        objects_file=objects_file,
        objects_list=objects_list,
        num_prompts=num_prompts,
        output=output,
        min_objects=min_objects,
        max_objects=max_objects,
        rpm=rpm,
        model=model,
        base_url=base_url,
        api_key=api_key,
        batch_size=batch_size,
    )
    result = prompt_gen_task(args)
    if result != 0:
        raise click.ClickException(
            f"Prompt generation task failed with exit code {result}"
        )


def prompt_gen_task(args) -> int:
    # Load objects
    objects = load_objects(args.objects_file, args.objects_list)
    if not objects:
        print("Error: No objects provided. Use --objects-file or --objects-list.")
        return 1

    # Validate min/max objects arguments
    if args.min_objects is not None or args.max_objects is not None:
        if args.min_objects is None or args.max_objects is None:
            print(
                "Error: Both --min-objects and --max-objects must be specified together."
            )
            return 1
        if args.min_objects > args.max_objects:
            print("Error: --min-objects cannot be greater than --max-objects.")
            return 1
        if args.min_objects < 1 or args.max_objects < 1:
            print("Error: --min-objects and --max-objects must be at least 1.")
            return 1
        if args.max_objects > len(objects):
            print(
                f"Error: --max-objects cannot be greater than the number of available objects ({len(objects)})."
            )
            return 1

    # Setup LLM client
    client = setup_llm_client(args.base_url, args.api_key)

    # Rate limiting setup
    rpm = args.rpm
    interval = 60 / rpm  # seconds between requests

    def generate_prompt(
        _: int,
        objects: list[str],
        min_objects: Optional[int],
        max_objects: Optional[int],
    ) -> Dict[str, Any]:
        if min_objects is not None and max_objects is not None:
            num = random.randint(min_objects, max_objects)
            selected_objects = random.sample(objects, num)
        else:
            selected_objects = objects
        objects_str = ", ".join(selected_objects)
        system_prompt = PROMPTGEN_SYS_PROMPT.format(objects=objects_str)
        response = rate_limited_call(
            client,
            model=args.model,
            messages=[{"role": "system", "content": system_prompt}],
            interval=interval,
        )
        result = validate_and_clean_prompt_gen_response(response)
        result = restructure_objects(result)
        return result

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
        for start in range(0, args.num_prompts, batch_size):
            batch_end = min(start + batch_size, args.num_prompts)
            batch_indices = range(start, batch_end)
            futures = [
                executor.submit(
                    generate_prompt, i, objects, args.min_objects, args.max_objects
                )
                for i in batch_indices
            ]
            batch_results = []
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Batch {start // batch_size + 1}",
            ):
                total_attempted += 1
                try:
                    result = future.result()
                    batch_results.append(result)
                    total_successful += 1
                except Exception as e:
                    print(f"Error in prompt generation: {e}")
                    total_failed += 1
            # Append batch results to file
            with open(args.output, "a") as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")

    print(f"Generated {total_successful} prompts to {args.output}")
    print("Prompt generation summary:")
    print(f"  Total attempted: {total_attempted}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    return 0


@cli.command("caption")
@click.option(
    "--input",
    type=str,
    required=True,
    help="Input directory containing images or path to a single image",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Output JSONL file path",
)
@add_common_llm_options
def caption(
    input: str,
    output: str,
    rpm: int,
    model: str,
    base_url: str,
    api_key: str,
    batch_size: int,
) -> None:
    """Generate image captions using OpenAI API."""
    from argparse import Namespace

    args = Namespace(
        input=input,
        output=output,
        rpm=rpm,
        model=model,
        base_url=base_url,
        api_key=api_key,
        batch_size=batch_size,
    )
    result = caption_task(args)
    if result != 0:
        raise click.ClickException(f"Caption task failed with exit code {result}")


def caption_task(args) -> int:
    """Run OpenAI-based image captioning task."""
    # Find image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []

    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input, ext.upper())))
    else:
        print(f"Error: Input path '{args.input}' does not exist.")
        return 1

    if not image_paths:
        print(f"No image files found in {args.input}")
        return 1

    # Setup LLM client
    try:
        client = setup_llm_client(args.base_url, args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Rate limiting setup
    rpm = args.rpm
    interval = 60 / rpm

    def process_image(img_path: str) -> Optional[Dict[str, Any]]:
        try:
            base64_image = encode_image_to_base64(img_path)
            response = rate_limited_call(
                client,
                model=args.model,
                messages=[
                    {"role": "system", "content": CAPTION_SYS_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                interval=interval,
            )
            return {"file_path": img_path, "caption": response}
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    # Clear output file
    with open(args.output, "w") as f:
        pass

    batch_size = args.batch_size
    total_attempted = 0
    total_failed = 0
    total_successful = 0

    cpu_count = os.cpu_count() or 1
    max_workers = min(rpm, batch_size, cpu_count)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, len(image_paths), batch_size):
            batch_end = min(start + batch_size, len(image_paths))
            batch_paths = image_paths[start:batch_end]
            futures = [
                executor.submit(process_image, img_path) for img_path in batch_paths
            ]
            batch_results = []
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Batch {start // batch_size + 1}",
            ):
                total_attempted += 1
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        total_successful += 1
                    else:
                        total_failed += 1
                except Exception as e:
                    print(f"Error in captioning: {e}")
                    total_failed += 1

            with open(args.output, "a") as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")

    print(f"Captioned {total_successful} images to {args.output}")
    print("Caption generation summary:")
    print(f"  Total attempted: {total_attempted}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    return 0


@cli.command("moondream-caption")
@click.option(
    "--input",
    type=str,
    required=True,
    help="Input directory containing images or path to a single image",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Output NDJSON file path",
)
@click.option(
    "--length",
    type=click.Choice(["short", "normal", "long"]),
    default="normal",
    help="Caption length (default: normal)",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Use local Moondream server at http://localhost:2020/v1",
)
@click.option(
    "--api-key",
    type=str,
    help="Moondream API key (default: from MOONDREAM_API_KEY env var)",
)
@click.option(
    "--rpm",
    type=int,
    default=60,
    help="Requests per minute limit (default: 60)",
)
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Batch size for processing (default: 10)",
)
@click.option(
    "--sequential",
    is_flag=True,
    default=False,
    help="Process images sequentially (useful for local models)",
)
def moondream_caption(
    input: str,
    output: str,
    length: str,
    local: bool,
    api_key: str,
    rpm: int,
    batch_size: int,
    sequential: bool,
) -> None:
    """Generate image captions using Moondream API."""
    from argparse import Namespace

    args = Namespace(
        input=input,
        output=output,
        length=length,
        local=local,
        api_key=api_key,
        rpm=rpm,
        batch_size=batch_size,
        sequential=sequential,
    )
    result = moondream_caption_task(args)
    if result != 0:
        raise click.ClickException(
            f"Moondream captioning task failed with exit code {result}"
        )


def moondream_caption_task(args) -> int:
    """Run Moondream image captioning task.

    Args:
        args: Namespace with input, output, length, local, api_key, rpm, batch_size, sequential

    Returns:
        0 on success, non-zero on failure.
    """
    # Find image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []

    if os.path.isfile(args.input):
        # Single image file
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        # Directory of images
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input, ext.upper())))
    else:
        print(f"Error: Input path '{args.input}' does not exist.")
        return 1

    if not image_paths:
        print(f"No image files found in {args.input}")
        return 1

    # Setup Moondream client
    try:
        client = setup_moondream_client(api_key=args.api_key, local=args.local)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Rate limiting setup
    rpm = args.rpm
    interval = 60 / rpm  # seconds between requests

    def process_image(img_path: str) -> Optional[Dict[str, Any]]:
        try:
            caption = rate_limited_caption(
                client=client,
                image_path=img_path,
                length=args.length,
                interval=interval,
            )
            return {"file_path": img_path, "caption": caption}
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    # Clear output file
    with open(args.output, "w") as f:
        pass

    batch_size = args.batch_size
    total_attempted = 0
    total_failed = 0
    total_successful = 0

    if args.sequential:
        # Sequential processing
        print("Running in sequential mode...")
        with open(args.output, "a") as f:
            for img_path in tqdm(image_paths, desc="Processing sequentially"):
                total_attempted += 1
                result = process_image(img_path)
                if result:
                    f.write(json.dumps(result) + "\n")
                    total_successful += 1
                else:
                    total_failed += 1
    else:
        # Parallel processing
        cpu_count = os.cpu_count() or 1
        max_workers = min(rpm, batch_size, cpu_count)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, len(image_paths), batch_size):
                batch_end = min(start + batch_size, len(image_paths))
                batch_paths = image_paths[start:batch_end]
                futures = [
                    executor.submit(process_image, img_path) for img_path in batch_paths
                ]
                batch_results = []
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Batch {start // batch_size + 1}",
                ):
                    total_attempted += 1
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                            total_successful += 1
                        else:
                            total_failed += 1
                    except Exception as e:
                        print(f"Error in image captioning: {e}")
                        total_failed += 1
                # Append batch results to file
                with open(args.output, "a") as f:
                    for result in batch_results:
                        f.write(json.dumps(result) + "\n")

    print(f"Captioned {total_successful} images to {args.output}")
    print("Moondream captioning summary:")
    print(f"  Total attempted: {total_attempted}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    return 0


def main() -> int:
    # Use Click's command invocation
    # The cli() function will handle command routing
    cli(prog_name="obh-generate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
