from __future__ import annotations

import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

from object_harvest.logging import get_logger
from object_harvest.reader import iter_images
from object_harvest.vlm import describe_objects_ndjson
from object_harvest.synthesis import synthesize_one_line
from object_harvest.utils.clients import AIClient
from object_harvest.detection import HFDataLoader, OVDModel, OVDMODEL
from object_harvest.writer import JSONDirWriter
from object_harvest.utils import (
    RateLimiter,
    append_and_maybe_flush_jsonl,
    flush_remaining_jsonl,
    update_tqdm_gpm,
)
from object_harvest.utils.paths import safe_stem
from object_harvest.utils.objects import (
    parse_objects_arg,
    load_describe_objects_map,
)
from object_harvest.utils.runs import resolve_run_dir_base
from tqdm import tqdm

logger = get_logger(__name__)


# ----------------------- Describe subcommand -----------------------
def _add_describe_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "describe",
        help="Suggest objects per image as NDJSON lines (object -> short description)",
        description='Generate NDJSON per image: each line is {"object": "object description"}. Includes people when present.',
    )
    p.add_argument(
        "--input", required=True, help="Input folder or a text file with paths/URLs"
    )
    p.add_argument(
        "--out", required=True, help="Output folder for per-image NDJSON files"
    )
    p.add_argument(
        "--model", default=os.getenv("OBJH_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    )
    p.add_argument(
        "--api-base",
        default=os.getenv("OBJH_API_BASE"),
        help="OpenAI-compatible base URL",
    )
    p.add_argument(
        "--max-workers", type=int, default=min(32, (os.cpu_count() or 4) * 5)
    )
    p.add_argument(
        "--rpm",
        type=int,
        default=int(os.getenv("OBJH_RPM", "0")),
        help="Requests per minute throttle (0=unlimited)",
    )
    p.add_argument(
        "--batch", type=int, default=0, help="Optional batch size; 0 processes all"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume into an existing run dir and skip images with existing JSON outputs",
    )
    p.set_defaults(command="describe")


def _process_one_describe(
    client: AIClient, item: dict, limiter: RateLimiter | None
) -> dict:
    path = item.get("path") or item.get("url")
    try:
        if limiter:
            limiter.acquire()
        ndjson_text = describe_objects_ndjson(client, item)
        return {"image": path, "ndjson": ndjson_text}
    except Exception as e:
        logger.error(f"failed processing {path}: {e}")
        return {"image": path, "ndjson": ""}


def _run_describe(args: argparse.Namespace) -> int:
    client = AIClient(model=args.model, base_url=args.api_base)
    items = list(iter_images(args.input, limit=args.batch if args.batch > 0 else None))
    logger.info(f"found {len(items)} images")

    # Resolve output base dir with resume support
    writer_base = resolve_run_dir_base(args.out, args.resume)
    if writer_base != args.out and args.resume:
        logger.info(f"resuming into {writer_base}")

    writer = JSONDirWriter(writer_base)
    existing_stems: set[str] = set()
    if args.resume:
        try:
            for name in os.listdir(writer.run_dir):
                if name.lower().endswith((".ndjson", ".json")):
                    existing_stems.add(os.path.splitext(name)[0])
        except FileNotFoundError:
            pass
    limiter = RateLimiter(rpm=args.rpm) if args.rpm and args.rpm > 0 else None

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = []
        for item in items:
            image_ref = item.get("path") or item.get("url") or "image"
            safe = safe_stem(image_ref)
            if args.resume and safe in existing_stems:
                continue
            futures.append(ex.submit(_process_one_describe, client, item, limiter))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            image_ref = rec.get("image") or "image"
            safe = safe_stem(image_ref)
            # Write NDJSON file per image
            nd = rec.get("ndjson", "") if isinstance(rec, dict) else ""
            writer.write_text(f"{safe}", nd, ext=".ndjson")

    total_written = (
        len(
            [
                n
                for n in os.listdir(writer.run_dir)
                if n.lower().endswith((".ndjson", ".json"))
            ]
        )
        if os.path.isdir(writer.run_dir)
        else 0
    )
    logger.info(f"done. outputs under {writer.run_dir}. files present: {total_written}")
    return 0


# ----------------------- Detect subcommand -----------------------
def _add_detect_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "detect",
        help="Open-vocabulary detection with HF models (GroundingDINO/LLMDet)",
        description=(
            "Perform object-grounded detections using Hugging Face models that support zero-shot OD. "
            "Can optionally read objects per-image from a previous describe run."
        ),
    )
    p.add_argument(
        "--input", required=True, help="Input folder or a text file with paths/URLs"
    )
    p.add_argument(
        "--out", required=True, help="Output folder for per-image detection JSON files"
    )
    p.add_argument(
        "--hf-model",
        default=None,
        help="Hugging Face model id (e.g., iSEE-Laboratory/llmdet_large)",
    )
    p.add_argument(
        "--hf-dataset",
        default=None,
        help="Optional Hugging Face dataset id to read inputs from (e.g., user/dataset). Overrides --input when set.",
    )
    p.add_argument(
        "--hf-dataset-split",
        default="train",
        help="Hugging Face dataset split to use (e.g., train, validation, test)",
    )
    p.add_argument(
        "--use-obj-desp",
        action="store_true",
        help="Use objects.description instead of objects.names when reading JSONL/HF datasets",
    )
    p.add_argument(
        "--use-describe",
        action="store_true",
        help="If set, looks for a sibling describe run dir (--input/../describe_*) and uses the per-image NDJSON files as object lists",
    )
    p.add_argument(
        "--objects",
        default=None,
        help="Either a path to a text file (one object per line) or a comma-separated list of object names",
    )
    p.add_argument(
        "--threshold", type=float, default=0.25, help="Score threshold for detections"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume into an existing run dir and skip images with existing JSON outputs",
    )
    p.set_defaults(command="detect")


"""helper functions moved into utils.objects and utils.detect_inputs"""


def _run_detect(args: argparse.Namespace) -> int:
    # Resolve run dir for outputs with resume support
    writer_base = resolve_run_dir_base(args.out, args.resume)
    if writer_base != args.out and args.resume:
        logger.info(f"resuming into {writer_base}")
    writer = JSONDirWriter(writer_base)
    
    #TODO: Currently only huggingface datasets supported
    #Check if hf-dataset is provided
    if getattr(args, "hf_dataset", None):
        data_loader = HFDataLoader(args.hf_dataset, use_desc=bool(getattr(args, "use_obj_desp", False)), split = args.hf_dataset_split)
        logger.info(f"loaded {len(data_loader)} items from HF dataset {args.hf_dataset}")
    else:
        logger.error("--detect currently only supports --hf-dataset mode")
        return 2

    # Initialize open-vocabulary detection model
    hf_model_id = args.hf_model or OVDMODEL
    ovd_model = OVDModel(hf_model_id, threshold=args.threshold)
    
    # Iterate over data_loader and perform detection
    for item in tqdm(data_loader, total=len(data_loader), desc="detect", unit="img"):
        rec = {
            "id": item["id"],
            "file_name": item["path"],
            "detections": [],
        }
        if len(item["prompt"]) > 0:
            detections = ovd_model(item["image"], item["prompt"])
            rec["detections"] = detections
        output_file_name = safe_stem(item["path"])
        writer.write(output_file_name, rec)

    logger.info(f"done. detection outputs under {writer.run_dir}")
    return 0


# ----------------------- Synthesis subcommand -----------------------
def _add_synthesis_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "synthesis",
        help="Generate a scene description using N objects from a given list",
        description="Use an LLM to synthesize a one-line image description that includes N objects from the provided list.",
    )
    p.add_argument(
        "--objects",
        default=None,
        help="Either a path to a text file (one object per line) or a comma-separated list of objects",
    )
    # Renamed for readability; keep --n as alias for backward-compat
    p.add_argument(
        "--num-objects",
        "--n",
        dest="num_objects",
        type=int,
        default=6,
        help="Number of objects to include in each description",
    )
    p.add_argument(
        "--model", default=os.getenv("OBJH_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    )
    p.add_argument("--api-base", default=os.getenv("OBJH_API_BASE"))
    p.add_argument(
        "--count", type=int, default=1, help="Number of synthetic samples to generate"
    )
    p.add_argument(
        "--rpm",
        type=int,
        default=int(os.getenv("OBJH_RPM", "0")),
        help="Requests per minute throttle (0=unlimited)",
    )
    p.add_argument(
        "--max-workers", type=int, default=min(32, (os.cpu_count() or 4) * 5)
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON/JSONL output; if omitted, prints to stdout",
    )
    p.add_argument(
        "--save-batch-size",
        type=int,
        default=0,
        help="For .jsonl outputs, flush results to disk in batches of this size (0 disables incremental batch writes)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=250,
        help="Max tokens for the LLM completion when synthesizing",
    )
    p.set_defaults(command="synthesis")


def _synthesize_text(
    model: str,
    base_url: str | None,
    objects: list[str],
    n: int,
    client: AIClient | None = None,
    max_tokens: int = 250,
) -> dict:
    return synthesize_one_line(
        objects, n, model, base_url, client=client, max_tokens=max_tokens
    )


def _run_synthesis(args: argparse.Namespace) -> int:
    objs: list[str] = parse_objects_arg(args.objects)
    if not objs:
        logger.error("Provide --objects as a file path or comma-separated list")
        return 2

    count = max(1, int(getattr(args, "count", 1)))
    num_objects = int(getattr(args, "num_objects", 6))

    limiter = RateLimiter(rpm=args.rpm) if getattr(args, "rpm", 0) else None
    save_batch_size = int(getattr(args, "save_batch_size", 0))

    # Create one shared LLM client to avoid opening too many connections/file descriptors
    shared_llm_client = AIClient(model=args.model, base_url=args.api_base)

    def one_sample() -> dict:
        if limiter:
            limiter.acquire()
        return _synthesize_text(
            args.model,
            args.api_base,
            objs,
            num_objects,
            client=shared_llm_client,
            max_tokens=int(getattr(args, "max_tokens", 250)),
        )

    results: list[dict] = []
    start = time.time()
    successes = 0
    failures = 0
    # Batch streaming only for .jsonl outputs when save_batch_size > 0
    streaming_jsonl = bool(
        args.out and args.out.lower().endswith(".jsonl") and save_batch_size > 0
    )
    fh = None
    write_lock = threading.Lock()
    current_batch: list[dict] = []

    if streaming_jsonl:
        out_dir = os.path.dirname(args.out) or "."
        os.makedirs(out_dir, exist_ok=True)
        fh = open(
            args.out, "a", encoding="utf-8"
        )  # append mode for incremental batches

    # Local adapters around utils helpers to keep call sites simple
    def _append_or_collect(rec: dict) -> None:
        if streaming_jsonl and fh is not None:
            append_and_maybe_flush_jsonl(
                rec, fh, write_lock, current_batch, save_batch_size
            )
        else:
            results.append(rec)

    def _flush_remaining() -> None:
        if streaming_jsonl and fh is not None:
            flush_remaining_jsonl(fh, write_lock, current_batch)

    def _update_progress(pbar: tqdm) -> None:
        update_tqdm_gpm(pbar, start, successes)

    try:
        if count == 1:
            with tqdm(total=1, desc="synthesis", unit="gen") as pbar:
                try:
                    rec = one_sample()
                    successes += 1
                    _append_or_collect(rec)
                except Exception as e:
                    failures += 1
                    logger.error(f"synthesis error: {e}")
                finally:
                    _update_progress(pbar)
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futures = [ex.submit(one_sample) for _ in range(count)]
                with tqdm(total=len(futures), desc="synthesis", unit="gen") as pbar:
                    for fut in as_completed(futures):
                        try:
                            rec = fut.result()
                            successes += 1
                            _append_or_collect(rec)
                        except Exception as e:
                            failures += 1
                            logger.error(f"synthesis error: {e}")
                        finally:
                            _update_progress(pbar)
        # After processing all, flush any remaining batch to disk
        _flush_remaining()
    finally:
        if fh is not None:
            fh.close()

    if failures:
        elapsed_min = max((time.time() - start) / 60.0, 1e-6)
        gpm = successes / elapsed_min
        logger.info(
            f"generated {successes}/{count} (failures={failures}), gpm={gpm:.1f}"
        )

    # Write output
    if args.out:
        if streaming_jsonl:
            # Already appended in batches
            logger.info(f"wrote (appended) {args.out}")
        else:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            if count == 1:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(results[0], f, ensure_ascii=False)
            else:
                if args.out.lower().endswith(".jsonl"):
                    with open(args.out, "w", encoding="utf-8") as f:
                        for rec in results:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    with open(args.out, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False)
            logger.info(f"wrote {args.out}")
    else:
        if count == 1:
            print(json.dumps(results[0], ensure_ascii=False))
        else:
            print(json.dumps(results, ensure_ascii=False))
    return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="object-harvest")
    sub = p.add_subparsers(dest="command")
    sub.required = False  # default to 'describe' for backward-compat

    _add_describe_parser(sub)
    _add_detect_parser(sub)
    _add_synthesis_parser(sub)

    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    cmd = getattr(args, "command", None)
    if cmd is None:
        # Default to 'describe' if no subcommand
        describe_argv = ["describe"] + ([] if argv is None else list(argv))
        return _run_describe(parse_args(describe_argv))

    if cmd == "describe":
        return _run_describe(args)
    if cmd == "detect":
        return _run_detect(args)
    if cmd == "synthesis":
        return _run_synthesis(args)
    logger.error(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
