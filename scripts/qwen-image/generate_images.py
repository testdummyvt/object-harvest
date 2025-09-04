"""Generate images from NDJSON synthesis output using Qwen-Image.

This CLI reads a JSONL/NDJSON file where each line contains a JSON object with
at least a "describe" key (as produced by src/object_harvest/synthesis.py).
For each line, it generates an image with Qwen-Image and saves it to the
specified output directory.

Example:
  uv run python scripts/qwen-image/generate_images.py \
	--input /path/to/coco_1.jsonl \
	--out ./generated

Optional knobs are available for model path, aspect ratio, steps, seed, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Tuple

from PIL import Image
from tqdm import tqdm

# Import local Qwen-Image helper and aspect catalog
from utils.qwen_image import QwenImage, ASPECT_RATIO_SIZES
from utils.prompt_enhance import enhance_prompt, SYS_PROMPT
from object_harvest.logging import get_logger

logger = get_logger(__name__)


def iter_descriptions(ndjson_path: Path) -> Iterator[Tuple[int, str]]:
    """Yield (index, description) pairs from an NDJSON file.

    Skips lines that cannot be parsed or lack a "describe" field.
    Index starts at 1 for stable file naming.
    """
    with ndjson_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            desc = obj.get("describe")
            if isinstance(desc, str) and desc.strip():
                yield i, desc.strip()


def count_descriptions(ndjson_path: Path) -> int:
    """Return the number of valid lines that contain a non-empty "describe" string."""
    count = 0
    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if (
                isinstance(obj, dict)
                and isinstance(obj.get("describe"), str)
                and obj["describe"].strip()
            ):
                count += 1
    return count


def save_image(
    img: Image.Image, out_dir: Path, stem: str, index: int, fmt: str
) -> Path:
    """Save PIL image as stem-<index>.<fmt> in out_dir; returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{stem}-{index:04d}.{fmt.lower()}"
    out_path = out_dir / filename
    save_kwargs = {}
    if fmt.lower() in {"jpg", "jpeg"}:
        save_kwargs = {"quality": 95}
    img.save(out_path, format=fmt.upper(), **save_kwargs)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate images from synthesis NDJSON using Qwen-Image"
    )
    p.add_argument(
        "--input", "-i", required=True, help="Path to NDJSON file produced by synthesis"
    )
    p.add_argument(
        "--out", "-o", required=True, help="Destination folder for generated images"
    )
    p.add_argument(
        "--model-path",
        default=None,
        help='HF model id or path (default: "Qwen/Qwen-Image")',
    )
    p.add_argument(
        "--aspect-ratio",
        choices=sorted(ASPECT_RATIO_SIZES.keys()),
        default=None,
        help="Aspect ratio for generation; defaults to a random supported ratio",
    )
    p.add_argument(
        "--steps", type=int, default=8, help="Number of inference steps (default: 8)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducibility (used when --no-randomize-seed)",
    )
    p.add_argument(
        "--randomize-seed",
        action="store_true",
        help="Use a random seed per image (default)",
    )
    p.add_argument(
        "--no-randomize-seed",
        dest="randomize_seed",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(randomize_seed=True)
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale (true_cfg_scale) for model",
    )
    p.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format (default: jpg)",
    )

    # Prompt enhancement options (optional)
    p.add_argument(
        "--enhance",
        action="store_true",
        help="Enhance each describe via LLM before generation",
    )
    p.add_argument(
        "--enhance-system-prompt",
        default=SYS_PROMPT,
        help="System prompt to steer enhancement (default: SYS_PROMPT)",
    )
    p.add_argument(
        "--enhance-model",
        default=None,
        help="Model to use for enhancement (default: ENV OBJH_MODEL)",
    )
    p.add_argument(
        "--enhance-base-url",
        default=None,
        help="Custom API base URL for enhancement (default: ENV OBJH_API_BASE)",
    )
    p.add_argument(
        "--enhance-temperature",
        type=float,
        default=0.3,
        help="Temperature for enhancement (default: 0.3)",
    )
    p.add_argument(
        "--enhance-max-tokens",
        type=int,
        default=256,
        help="Max tokens for enhancement (default: 256)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ndjson_path = Path(args.input)
    out_dir = Path(args.out)

    if not ndjson_path.exists():
        raise FileNotFoundError(f"Input NDJSON not found: {ndjson_path}")

    # Use input file name stem for output file prefix
    stem = ndjson_path.stem or "images"

    generator = QwenImage(model_path=args.model_path)

    if args.enhance:
        logger.info(
            "Prompt enhancement enabled (temp=%s, max_tokens=%s)",
            args.enhance_temperature,
            args.enhance_max_tokens,
        )

    count = 0
    total = count_descriptions(ndjson_path)
    for idx, describe in tqdm(
        iter_descriptions(ndjson_path), total=total, desc="Generating", unit="img"
    ):
        prompt_text = describe
        if args.enhance:
            try:
                prompt_text = enhance_prompt(
                    user_prompt=describe,
                    system_prompt=args.enhance_system_prompt,
                    model=args.enhance_model,
                    base_url=args.enhance_base_url,
                    temperature=args.enhance_temperature,
                    max_tokens=args.enhance_max_tokens,
                )
            except Exception as e:
                logger.warning("Enhancement failed for index %s: %s", idx, e)
                prompt_text = describe

        img = generator(
            # Replace describe with enhanced text if enhancement is enabled.
            prompt=prompt_text,
            negative_prompt=" ",
            aspect_ratio=args.aspect_ratio,
            num_inference_steps=args.steps,
            seed=args.seed,
            randomize_seed=args.randomize_seed,
            guidance_scale=args.guidance_scale,
        )
        path = save_image(img, out_dir, stem, idx, args.format)
        count += 1
        logger.info(f"Saved: {path}")

    if count == 0:
        logger.warning("No valid describe entries found in input. Nothing generated.")
    else:
        logger.info(f"Done. Generated {count} image(s) into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
