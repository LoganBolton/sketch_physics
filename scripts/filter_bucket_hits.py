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
    line_counts = {1: 0, 2: 0, 3: 0}  # Track scenes by number of lines

    for run_dir in sorted(source_dir.glob("run_*")):
        metadata_file = run_dir / args.metadata_file
        if not metadata_file.exists():
            print(f"Warning: {metadata_file} not found, skipping")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        bucket_hit = metadata.get("simulation", {}).get("bucket_hit")
        bounce_detected = metadata.get("simulation", {}).get("bounce_detected")
        if bucket_hit is not None and not bounce_detected:
            new_run_dir = target_dir / run_dir.name
            # Remove existing directory if it exists
            if new_run_dir.exists():
                shutil.rmtree(new_run_dir)
            shutil.copytree(run_dir, new_run_dir)

            # Track line count
            num_lines = metadata.get("simulation", {}).get("num_lines")
            if num_lines in line_counts:
                line_counts[num_lines] += 1

            print(f"Copied {run_dir.name} (bucket {bucket_hit}, {num_lines} lines)")
            copied += 1
        else:
            skipped += 1

    print(f"\nCopied {copied} scenes with bucket hits")
    print(f"Skipped {skipped} scenes without bucket hits")
    print(f"\nScenes by number of lines:")
    print(f"  1 line:  {line_counts[1]} scenes")
    print(f"  2 lines: {line_counts[2]} scenes")
    print(f"  3 lines: {line_counts[3]} scenes")


if __name__ == "__main__":
    main()
