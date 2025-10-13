"""Filter random scene outputs to only keep scenes where the ball hits a bucket."""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter scene outputs by bucket hits")
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Source directory with run_* subdirectories")
    parser.add_argument("--target-dir", type=str, required=True,
                        help="Target directory for filtered scenes")
    parser.add_argument("--metadata-file", type=str, default="random_scene_metadata.json",
                        help="Metadata filename (default: random_scene_metadata.json)")
    return parser.parse_args()


def main():
    args = parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for run_dir in sorted(source_dir.glob("run_*")):
        metadata_file = run_dir / args.metadata_file
        if not metadata_file.exists():
            print(f"Warning: {metadata_file} not found, skipping")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        bucket_hit = metadata.get("simulation", {}).get("bucket_hit")
        if bucket_hit is not None:
            new_run_dir = target_dir / run_dir.name
            shutil.copytree(run_dir, new_run_dir)
            print(f"Copied {run_dir.name} (bucket {bucket_hit})")
            copied += 1
        else:
            skipped += 1

    print(f"\nCopied {copied} scenes with bucket hits")
    print(f"Skipped {skipped} scenes without bucket hits")


if __name__ == "__main__":
    main()
