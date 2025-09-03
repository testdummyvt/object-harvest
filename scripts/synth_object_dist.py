#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_objects_from_jsonl(jsonl_path):
    """
    Loads all objects (use the keys of each object dict) from a synthesis output JSONL file.
    Each line: { "describe": str, "objects": [ {object: description}, ... ] }
    We collect the dict keys (object names), not the values.
    """
    all_objects = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                objects = data.get("objects", [])
                for obj in objects:
                    # Use the keys of the object dict (object names), not their values/descriptions
                    for key in obj.keys():
                        if isinstance(key, str):
                            k = key.strip().lower()
                            if k:
                                all_objects.append(k)
            except Exception as e:
                print(f"Warning: failed to parse line: {e}")
    return all_objects


def plot_distribution(counter, top_n=30, out_file=None):
    labels, counts = zip(*counter.most_common(top_n)) if counter else ([], [])
    plt.figure(figsize=(12, 6))
    if labels:
        plt.barh(labels[::-1], counts[::-1])
    plt.xlabel("Frequency")
    plt.title(f"Top {top_n} Synthesized Object Descriptions")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        print(f"Plot saved to {out_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot distribution of generated object descriptions from synthesis JSONL output."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Synthesis output JSONL file."
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=30,
        help="Show top N descriptions (default: 30).",
    )
    parser.add_argument(
        "--out",
        "-o",
        help="Output file for plot (e.g., dist.png). If not set, shows interactively.",
    )
    args = parser.parse_args()

    print(f"Loading synthesis outputs from: {args.input}")
    objects = load_objects_from_jsonl(args.input)
    print(f"Loaded {len(objects)} object descriptions.")

    counter = Counter(objects)
    print(f"Found {len(counter)} unique descriptions.")

    plot_distribution(counter, top_n=args.top, out_file=args.out)


if __name__ == "__main__":
    main()
