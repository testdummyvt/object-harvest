import os
from typing import Optional, Dict, Any
import random
import click
import json
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from obh.utils import load_objects, setup_llm_client, rate_limited_call
from obh.utils.prompts import PROMPTGEN_SYS_PROMPT, QWEN_T2I_SYS_PROMPT, MAGIC_PROMPT_EN
from obh.utils.validation import validate_and_clean_prompt_gen_response, restructure_objects


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
        help="Batch size for processing (default: 100)",
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
def prompt_gen(objects_file: str, objects_list: str, num_prompts: int, output: str, 
               min_objects: int, max_objects: int, rpm: int, model: str, 
               base_url: str, api_key: str, batch_size: int) -> None:
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
        batch_size=batch_size
    )
    result = prompt_gen_task(args)
    if result != 0:
        raise click.ClickException(f"Prompt generation task failed with exit code {result}")


@cli.command()
@click.option("--input", type=str, required=True, help="Input NDJSON file")
@click.option("--output", type=str, required=True, help="Output directory to save generated images and metadata")
@click.option("--model-path", type=str, default="Qwen/Qwen-Image", help="Hugging Face model path (default: Qwen/Qwen-Image)")
@click.option("--input-prompt-field", type=str, default="prompt", help="Field in input NDJSON to use as prompt (default: prompt)")
@click.option("--aspect-ratio", type=str, help="Aspect ratio from ASPECT_RATIO_SIZES (optional)")
@click.option("--num-inference-steps", type=int, default=8, help="Number of inference steps (default: 8)")
@click.option("--steps", type=int, default=1, help="Number of images to generate per prompt (default: 1)")
@click.option("--seed", type=int, default=0, help="Seed for reproducibility (default: 0)")
@click.option("--randomize-seed/--no-randomize-seed", default=True, help="Randomize seed (default: True)")
@click.option("--guidance-scale", type=float, default=1.0, help="Guidance scale (default: 1.0)")
@click.option("--format", type=click.Choice(["png", "jpeg"]), default="jpeg", help="Image format (default: jpeg)")
def image_gen(input: str, output: str, model_path: str, input_prompt_field: str, 
              aspect_ratio: str, num_inference_steps: int, steps: int, seed: int, 
              randomize_seed: bool, guidance_scale: float, format: str) -> None:
    """Generate images using Qwen-Image."""
    # Create args namespace to maintain compatibility with existing image_gen_task function
    from argparse import Namespace
    args = Namespace(
        input=input,
        output=output,
        model_path=model_path,
        input_prompt_field=input_prompt_field,
        aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps,
        steps=steps,
        seed=seed,
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        format=format
    )
    result = image_gen_task(args)
    if result != 0:
        raise click.ClickException(f"Image generation task failed with exit code {result}")


@cli.command()
@click.option(
    "--input",
    type=str,
    required=True,
    help="Input NDJSON file generated by prompt-gen",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Output NDJSON file path",
)
@add_common_llm_options
def prompt_enhance(input: str, output: str, rpm: int, model: str, 
                   base_url: str, api_key: str, batch_size: int) -> None:
    """Enhance prompts using Qwen T2I prompt optimizer."""
    # Create args namespace to maintain compatibility with existing prompt_enhance_task function
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
    result = prompt_enhance_task(args)
    if result != 0:
        raise click.ClickException(f"Prompt enhancement task failed with exit code {result}")


def prompt_gen_task(args) -> int:
    # Load objects
    objects = load_objects(args.objects_file, args.objects_list)
    if not objects:
        print("Error: No objects provided. Use --objects-file or --objects-list.")
        return 1

    # Validate min/max objects arguments
    if args.min_objects is not None or args.max_objects is not None:
        if args.min_objects is None or args.max_objects is None:
            print("Error: Both --min-objects and --max-objects must be specified together.")
            return 1
        if args.min_objects > args.max_objects:
            print("Error: --min-objects cannot be greater than --max-objects.")
            return 1
        if args.min_objects < 1 or args.max_objects < 1:
            print("Error: --min-objects and --max-objects must be at least 1.")
            return 1
        if args.max_objects > len(objects):
            print(f"Error: --max-objects cannot be greater than the number of available objects ({len(objects)}).")
            return 1

    # Setup LLM client
    client = setup_llm_client(args.base_url, args.api_key)

    # Rate limiting setup
    rpm = args.rpm
    interval = 60 / rpm  # seconds between requests

    def generate_prompt(_: int, objects: list[str], min_objects: Optional[int], max_objects: Optional[int]) -> Dict[str, Any]:
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
            futures = [executor.submit(generate_prompt, i, objects, args.min_objects, args.max_objects) for i in batch_indices]
            batch_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {start//batch_size + 1}"):
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


def image_gen_task(args) -> int:
    # Validate aspect_ratio if provided
    if args.aspect_ratio and args.aspect_ratio not in ASPECT_RATIO_SIZES:
        print(f"Error: Invalid aspect ratio '{args.aspect_ratio}'. Valid options: {list(ASPECT_RATIO_SIZES.keys())}")
        return 1

    # Create output directories
    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Initialize QwenImage
    generator = QwenImage(model_path=args.model_path)

    # Read input NDJSON
    input_data = []
    try:
        with open(args.input, "r") as f:
            for line in f:
                if line.strip():
                    input_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return 1

    # Prepare metadata file
    metadata_path = os.path.join(args.output, "metadata.ndjson")
    with open(metadata_path, "w") as f:
        pass  # clear

    total_images = 0
    for idx, entry in enumerate(tqdm(input_data, desc="Processing prompts")):
        # Get prompt
        prompt = entry.get(args.input_prompt_field)
        if prompt is None:
            prompt = entry.get("describe")
        if prompt is None:
            print(f"Warning: No '{args.input_prompt_field}' or 'describe' field in entry {idx}, skipping.")
            continue

        # Generate steps images
        for step in range(args.steps):
            if args.randomize_seed:
                seed = random.randint(0, MAX_SEED)
                print(f"Using random seed {seed} for entry {idx}, step {step}")
            else:
                seed = args.seed + step  # vary per step for reproducibility

            # Generate image
            image = generator(
                prompt=prompt,
                aspect_ratio=args.aspect_ratio,
                num_inference_steps=args.num_inference_steps,
                seed=seed,
                randomize_seed=False,  # we handle seed here
                guidance_scale=args.guidance_scale,
            )

            # Generate uuid
            img_uuid = str(uuid.uuid4())

            # Filename
            filename = f"{idx}_{img_uuid}.{args.format}"

            img_path = os.path.join(images_dir, filename)

            # Save image
            image.save(img_path, args.format.upper())

            # Metadata entry
            meta_entry = {
                "image_path": f"images/{filename}",
            }
            if "prompt" in entry:
                meta_entry["prompt"] = entry["prompt"]
            if "describe" in entry:
                meta_entry["describe"] = entry["describe"]
            if "objects" in entry:
                meta_entry["objects"] = entry["objects"]

            # Append to metadata
            with open(metadata_path, "a") as f:
                f.write(json.dumps(meta_entry) + "\n")

            total_images += 1

    print(f"Generated {total_images} images to {args.output}")
    return 0


def prompt_enhance_task(args) -> int:
    # Setup LLM client
    client = setup_llm_client(args.base_url, args.api_key)

    # Rate limiting setup
    rpm = args.rpm
    interval = 60 / rpm  # seconds between requests

    # Read input NDJSON file
    input_data = []
    try:
        with open(args.input, "r") as f:
            for line in f:
                if line.strip():
                    input_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return 1

    def enhance_prompt(entry: dict) -> dict:
        # Get the description to enhance
        description = entry.get("describe", "")
        
        # Call the LLM to enhance the prompt using QWEN_T2I_SYS_PROMPT as system prompt
        # and the description as the user input
        enhanced = rate_limited_call(
            client,
            model=args.model,
            messages=[
                {"role": "system", "content": QWEN_T2I_SYS_PROMPT},
                {"role": "user", "content": description + " " + MAGIC_PROMPT_EN},
            ],
            interval=interval,
        )
        
        # Create a new entry with the enhanced prompt
        new_entry = entry.copy()
        new_entry["prompt"] = enhanced
        return new_entry

    # Clear output file
    with open(args.output, "w") as f:
        pass

    batch_size = args.batch_size
    total_processed = 0
    # Ensure rpm and batch_size are not None before using min
    cpu_count = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    max_workers = min(rpm, batch_size, cpu_count)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, len(input_data), batch_size):
            batch = input_data[start:start + batch_size]
            futures = [executor.submit(enhance_prompt, entry) for entry in batch]
            batch_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {start//batch_size + 1}"):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error in prompt enhancement: {e}")
            # Append batch results to file
            with open(args.output, "a") as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")
            total_processed += len(batch_results)

    print(f"Enhanced {total_processed} prompts to {args.output}")
    return 0


def main() -> int:
    # Use Click's command invocation
    # The cli() function will handle command routing
    cli(prog_name='obh-generate')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
