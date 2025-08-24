from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

from object_harvest.logging import get_logger
from object_harvest.reader import iter_images
from object_harvest.vlm import VLMClient, describe_and_list
from object_harvest.synthesis import synthesize_one_line
from object_harvest.detection import run_gdino_detection, run_vlm_detection
from object_harvest.writer import JSONDirWriter
from object_harvest.utils import RateLimiter
from tqdm import tqdm

logger = get_logger(__name__)


def _safe_stem(image_ref: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(image_ref)
    if parsed.scheme in ("http", "https"):
        stem = os.path.basename(parsed.path) or parsed.netloc
    else:
        stem = os.path.splitext(os.path.basename(image_ref))[0] or "image"
    return stem.replace("/", "_").replace("\\", "_") or "image"


# ----------------------- Describe subcommand -----------------------
def _add_describe_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "describe",
        help="Describe images and list objects (per-image JSON outputs)",
        description="Extract image descriptions and object lists from images using VLMs; writes one JSON per image in a run folder.",
    )
    p.add_argument("--input", required=True, help="Input folder or a text file with paths/URLs")
    p.add_argument("--out", required=True, help="Output folder for per-image JSON files")
    p.add_argument("--model", default=os.getenv("OBJH_MODEL", "qwen/qwen2.5-vl-72b-instruct"))
    p.add_argument("--api-base", default=os.getenv("OBJH_API_BASE"), help="OpenAI-compatible base URL")
    p.add_argument("--max-workers", type=int, default=min(32, (os.cpu_count() or 4) * 5))
    p.add_argument("--rpm", type=int, default=int(os.getenv("OBJH_RPM", "0")), help="Requests per minute throttle (0=unlimited)")
    p.add_argument("--batch", type=int, default=0, help="Optional batch size; 0 processes all")
    p.add_argument("--resume", action="store_true", help="Resume into an existing run dir and skip images with existing JSON outputs")
    p.set_defaults(command="describe")


def _process_one_describe(client: VLMClient, item: dict, limiter: RateLimiter | None) -> dict:
    path = item.get("path") or item.get("url")
    try:
        if limiter:
            limiter.acquire()
        result = describe_and_list(client, item)
        return {
            "image": path,
            "description": result.get("description"),
            "objects": result.get("objects", []),
        }
    except Exception as e:
        logger.error(f"failed processing {path}: {e}")
        return {"image": path, "description": None, "objects": []}


def _run_describe(args: argparse.Namespace) -> int:
    client = VLMClient(model=args.model, base_url=args.api_base)
    items = list(iter_images(args.input, limit=args.batch if args.batch > 0 else None))
    logger.info(f"found {len(items)} images")

    # Determine run directory (resume support)
    writer_base = args.out
    if args.resume:
        try:
            if os.path.isdir(args.out) and not os.path.basename(args.out).startswith("run-"):
                run_dirs = [
                    os.path.join(args.out, d)
                    for d in os.listdir(args.out)
                    if d.startswith("run-") and os.path.isdir(os.path.join(args.out, d))
                ]
                if run_dirs:
                    writer_base = max(run_dirs, key=os.path.getmtime)
                    logger.info(f"resuming into {writer_base}")
        except Exception:
            pass

    writer = JSONDirWriter(writer_base)
    existing_stems: set[str] = set()
    if args.resume:
        try:
            for name in os.listdir(writer.run_dir):
                if name.lower().endswith(".json"):
                    existing_stems.add(os.path.splitext(name)[0])
        except FileNotFoundError:
            pass
    limiter = RateLimiter(rpm=args.rpm) if args.rpm and args.rpm > 0 else None

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = []
        for item in items:
            image_ref = item.get("path") or item.get("url") or "image"
            safe = _safe_stem(image_ref)
            if args.resume and safe in existing_stems:
                continue
            futures.append(ex.submit(_process_one_describe, client, item, limiter))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            image_ref = rec.get("image") or "image"
            safe = _safe_stem(image_ref)
            writer.write(f"{safe}", rec)

    total_written = (
        len([n for n in os.listdir(writer.run_dir) if n.lower().endswith(".json")])
        if os.path.isdir(writer.run_dir)
        else 0
    )
    logger.info(f"done. outputs under {writer.run_dir}. files present: {total_written}")
    return 0


# ----------------------- Detect subcommand -----------------------
def _add_detect_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "detect",
        help="Open-vocabulary detection using models like GroundingDINO or LLMDet",
        description=(
            "Perform description- or object-based detections. Can optionally read objects per-image from a previous describe run."
        ),
    )
    p.add_argument("--input", required=True, help="Input folder or a text file with paths/URLs")
    p.add_argument("--out", required=True, help="Output folder for per-image detection JSON files")
    p.add_argument("--backend", choices=["gdino", "vlm"], default="gdino", help="Detection backend")
    p.add_argument("--hf-model", default=None, help="Hugging Face model id for the chosen backend")
    p.add_argument("--model", default=os.getenv("OBJH_MODEL", "qwen/qwen2.5-vl-72b-instruct"), help="VLM model name for --backend vlm")
    p.add_argument("--api-base", default=os.getenv("OBJH_API_BASE"), help="OpenAI-compatible base URL for --backend vlm")
    p.add_argument("--from-describe", default=None, help="Path to a describe run-* directory to read objects per image")
    p.add_argument("--objects-file", default=None, help="Text file with one object per line to detect")
    p.add_argument("--objects", default=None, help="Comma-separated object names to detect (used if no file provided)")
    p.add_argument("--text", default=None, help="Free-form object description prompt for GroundingDINO (optional)")
    p.add_argument("--threshold", type=float, default=0.25, help="Score threshold for detections")
    p.add_argument("--max-workers", type=int, default=min(32, (os.cpu_count() or 4) * 5))
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="Resume into an existing run dir and skip images with existing JSON outputs")
    p.set_defaults(command="detect")


def _load_objects_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _load_describe_objects_map(run_dir: str) -> dict[str, list[str]]:
    """Map safe stem -> objects[] from a describe run dir."""
    mapping: dict[str, list[str]] = {}
    for name in os.listdir(run_dir):
        if not name.lower().endswith(".json"):
            continue
        stem = os.path.splitext(name)[0]
        try:
            with open(os.path.join(run_dir, name), "r", encoding="utf-8") as f:
                data = json.load(f)
            objs = data.get("objects") or []
            if isinstance(objs, list):
                mapping[stem] = [str(o) for o in objs]
        except Exception:
            continue
    return mapping


def _detect_backend_available(backend: str) -> bool:
    if backend == "gdino":
        import importlib.util

        return importlib.util.find_spec("transformers") is not None
    if backend == "vlm":
        return True
    return False


def _run_detection_on_item(
    backend: str,
    hf_model: str | None,
    threshold: float,
    item: dict,
    labels: list[str] | None,
    text_prompt: str | None,
) -> dict:
    """Stub detection runner. Emits empty detections if backend not available."""
    path = item.get("path") or item.get("url")
    detections: list[dict] = []
    if _detect_backend_available(backend):
        # TODO: implement model loading/inference for selected backend
        pass
    else:
        logger.warning(
            f"backend '{backend}' not available. Install transformers/torch and configure model (hf-model). Emitting empty detections."
        )
    return {"image": path, "detections": detections}


def _run_detect(args: argparse.Namespace) -> int:
    # Resolve run dir for outputs with resume support
    writer_base = args.out
    if args.resume:
        try:
            if os.path.isdir(args.out) and not os.path.basename(args.out).startswith("run-"):
                run_dirs = [
                    os.path.join(args.out, d)
                    for d in os.listdir(args.out)
                    if d.startswith("run-") and os.path.isdir(os.path.join(args.out, d))
                ]
                if run_dirs:
                    writer_base = max(run_dirs, key=os.path.getmtime)
                    logger.info(f"resuming into {writer_base}")
        except Exception:
            pass
    writer = JSONDirWriter(writer_base)

    existing_stems: set[str] = set()
    if args.resume:
        try:
            for name in os.listdir(writer.run_dir):
                if name.lower().endswith(".json"):
                    existing_stems.add(os.path.splitext(name)[0])
        except FileNotFoundError:
            pass

    # Objects source
    global_labels: list[str] | None = None
    if args.objects_file:
        global_labels = _load_objects_from_file(args.objects_file)
    elif args.objects:
        global_labels = [s.strip() for s in args.objects.split(",") if s.strip()]

    per_image_labels: dict[str, list[str]] = {}
    if args.from_describe:
        # if path points to parent folder, auto-pick latest run
        src = args.from_describe
        if os.path.isdir(src) and not os.path.basename(src).startswith("run-"):
            try:
                run_dirs = [
                    os.path.join(src, d)
                    for d in os.listdir(src)
                    if d.startswith("run-") and os.path.isdir(os.path.join(src, d))
                ]
                if run_dirs:
                    src = max(run_dirs, key=os.path.getmtime)
            except Exception:
                pass
        per_image_labels = _load_describe_objects_map(src)

    items = list(iter_images(args.input, limit=args.batch if args.batch > 0 else None))
    logger.info(f"found {len(items)} images")

    def labels_for_item(item: dict) -> list[str] | None:
        image_ref = item.get("path") or item.get("url") or "image"
        stem = _safe_stem(image_ref)
        return per_image_labels.get(stem) or global_labels

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = []
        future_to_image: dict = {}
        client_vlm = VLMClient(model=args.model, base_url=args.api_base) if args.backend == "vlm" else None

        for item in items:
            image_ref = item.get("path") or item.get("url") or "image"
            stem = _safe_stem(image_ref)
            if args.resume and stem in existing_stems:
                continue

            labels = labels_for_item(item)
            if args.backend == "gdino":
                fut = ex.submit(
                    run_gdino_detection,
                    item,
                    labels,
                    args.threshold,
                    args.hf_model,
                    args.text,
                )
            elif args.backend == "vlm":
                fut = ex.submit(
                    run_vlm_detection,
                    client_vlm,  # type: ignore[arg-type]
                    item,
                    labels,
                )
            else:
                continue
            futures.append(fut)
            future_to_image[fut] = image_ref

        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            image_ref = future_to_image.get(fut, "image")
            # Standardize output shape
            out = rec if (isinstance(rec, dict) and "image" in rec and "detections" in rec) else {"image": image_ref, "detections": rec or []}
            safe = _safe_stem(out.get("image") or "image")
            writer.write(f"{safe}", out)

    logger.info(f"done. detection outputs under {writer.run_dir}")
    return 0


# ----------------------- Synthesis subcommand -----------------------
def _add_synthesis_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "synthesis",
        help="Generate a scene description using N objects from a given list",
        description="Use an LLM to synthesize a one-line image description that includes N objects from the provided list.",
    )
    p.add_argument("--objects-file", default=None, help="Text file with one object per line")
    p.add_argument("--objects", default=None, help="Comma-separated list of objects")
    p.add_argument("--n", type=int, default=6, help="Number of objects to include in the description")
    p.add_argument("--model", default=os.getenv("OBJH_MODEL", "qwen/qwen2.5-vl-72b-instruct"))
    p.add_argument("--api-base", default=os.getenv("OBJH_API_BASE"))
    p.add_argument("--out", default=None, help="Optional file path to write JSON output; if omitted, prints to stdout")
    p.set_defaults(command="synthesis")


def _synthesize_text(model: str, base_url: str | None, objects: list[str], n: int) -> dict:
    return synthesize_one_line(objects, n, model, base_url)


def _run_synthesis(args: argparse.Namespace) -> int:
    objs: list[str] = []
    if args.objects_file:
        objs = _load_objects_from_file(args.objects_file)
    elif args.objects:
        objs = [s.strip() for s in args.objects.split(",") if s.strip()]
    else:
        logger.error("Provide --objects-file or --objects")
        return 2
    if not objs:
        logger.error("No objects provided after parsing inputs")
        return 2

    rec = _synthesize_text(args.model, args.api_base, objs, args.n)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        logger.info(f"wrote {args.out}")
    else:
        print(json.dumps(rec, ensure_ascii=False))
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
        return _run_describe(parse_args(["describe", *([] if argv is None else list(argv))]))

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
