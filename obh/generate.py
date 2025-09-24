import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


from obh.utils import load_objects, setup_llm_client, rate_limited_call

PROMPTGEN_SYS_PROMPT = (
    "You are a concise captioning assistant. Using ONLY the following objects — each supplied with a short visual descriptor (color, size, texture, or lighting) — write a vivid one-line scene description that naturally includes all of them without using a list format: {objects}\n\n"
    'Format requirement for {objects}: provide a comma-separated list where each entry is "object_name — short visual descriptor" (for example: rose — deep crimson, velvety petals; lantern — brass, warm glow). The assistant must use those visual descriptors when composing the scene.\n\n'
    "Avoid meta phrases like 'in this image' or 'this picture shows'.\n\n"
    "Then output STRICT JSON with exactly two keys and valid JSON syntax (no extra text outside the JSON):\n\n"
    "{{\n"
    '  "describe": "<the one-line description>",\n'
    '  "objects": [\n'
    '    {{"<object_1>": "<the exact object_1 phrasing as used within the description>"}},\n'
    '    {{"<object_2>": "<the exact object_2 phrasing as used within the description>"}}\n'
    "  ]\n"
    "}}\n\n"
    "Rules:\n"
    '- Use the exact object names (the part before the "—") from the provided list as JSON keys.\n'
    "- The object values in the JSON must match exactly how each object (including its short visual descriptor) appears in the main description.\n"
    "- Do not add or remove objects; include every provided object exactly once.\n"
    '- If a person is present among the objects, explicitly mention the type of clothing that person is wearing (brief, descriptive phrase) and, if possible, name any accessories they are wearing (e.g., "linen shirt, cuffed jeans; leather satchel, gold hoop earrings"). Clothing and accessories must appear naturally within the one-line description and be reflected exactly in the corresponding object value in the JSON.\n'
    "- Keep object descriptions consistent with the wording in the main description.\n"
    "- Output JSON only (no backticks, no explanations, no extra characters)."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object Harvest Generator")
    subparsers = parser.add_subparsers(dest="task", help="Available tasks")

    # Prompt-gen subcommand
    prompt_parser = subparsers.add_parser("prompt-gen", help="Generate prompts using LLM")
    prompt_parser.add_argument(
        "--objects-file",
        type=str,
        help="Path to text file with objects (one per line, format: 'object — descriptor')",
    )
    prompt_parser.add_argument(
        "--objects-list",
        type=str,
        help="Comma-separated list of objects (format: 'object — descriptor')",
    )
    prompt_parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of prompts to generate",
    )
    prompt_parser.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="Requests per minute limit (default: 60)",
    )
    prompt_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output NDJSON file path",
    )
    prompt_parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="LLM model to use (default: openai/gpt-4o)",
    )
    prompt_parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for LLM API (default: OpenRouter)",
    )
    prompt_parser.add_argument(
        "--api-key",
        type=str,
        help="API key (default: from OPENROUTER_API_KEY env var)",
    )

    # Image-gen subcommand (placeholder)
    image_parser = subparsers.add_parser("image-gen", help="Generate images (not implemented)")
    image_parser.add_argument("--input", type=str, help="Input NDJSON file")

    return parser.parse_args()


def prompt_gen_task(args: argparse.Namespace) -> int:
    # Load objects
    objects = load_objects(args.objects_file, args.objects_list)
    if not objects:
        print("Error: No objects provided. Use --objects-file or --objects-list.")
        return 1

    # Setup LLM client
    client = setup_llm_client(args.base_url, args.api_key)

    # Prepare system prompt
    objects_str = ", ".join(objects)
    system_prompt = PROMPTGEN_SYS_PROMPT.format(objects=objects_str)

    # Rate limiting setup
    rpm = args.rpm
    interval = 60 / rpm  # seconds between requests

    def generate_prompt(i: int) -> str:
        return rate_limited_call(
            client,
            model=args.model,
            messages=[{"role": "system", "content": system_prompt}],
            interval=interval,
        )

    # Multi-threaded generation
    with ThreadPoolExecutor(max_workers=min(rpm, args.num_prompts)) as executor:
        futures = [executor.submit(generate_prompt, i) for i in range(args.num_prompts)]
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in prompt generation: {e}")

    # Write to NDJSON
    with open(args.output, "w") as f:
        for result in results:
            f.write(result + "\n")

    print(f"Generated {len(results)} prompts to {args.output}")
    return 0


def image_gen_task(args: argparse.Namespace) -> int:
    raise NotImplementedError("Image generation not yet implemented")


def main() -> int:
    args = parse_args()
    if args.task == "prompt-gen":
        return prompt_gen_task(args)
    elif args.task == "image-gen":
        return image_gen_task(args)
    else:
        print("Error: No task specified. Use 'prompt-gen' or 'image-gen'.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())