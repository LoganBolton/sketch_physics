"""Generate an HTML summary of all runs in a directory."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML summary of random scene runs.")
    parser.add_argument("input_dir", type=pathlib.Path,
                        help="Directory containing run subdirectories")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Output HTML file path (default: input_dir/summary.html)")
    return parser.parse_args()


def collect_run_data(input_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Collect data from all runs in the directory."""
    runs = []

    # Check if there are run_XXX subdirectories or if files are directly in input_dir
    run_dirs = sorted(input_dir.glob("run_*"))

    if run_dirs:
        # Multiple runs in subdirectories
        for run_dir in run_dirs:
            metadata_path = run_dir / "random_scene_metadata.json"
            if not metadata_path.exists():
                continue

            with metadata_path.open("r") as f:
                metadata = json.load(f)

            start_image = run_dir / "random_scene_start.png"
            final_image = run_dir / "random_scene_final.png"

            if not start_image.exists() or not final_image.exists():
                continue

            runs.append({
                "name": run_dir.name,
                "start_image": start_image.relative_to(input_dir),
                "final_image": final_image.relative_to(input_dir),
                "bucket_hit": metadata["simulation"].get("bucket_hit"),
                "num_lines": metadata["simulation"].get("num_lines"),
                "start_ball_x": metadata["simulation"].get("start_ball_x"),
                "final_ball_x": metadata["simulation"].get("final_ball_x"),
                "final_ball_y": metadata["simulation"].get("final_ball_y"),
                "seed": metadata.get("seed"),
            })
    else:
        # Single run directly in the directory
        metadata_path = input_dir / "random_scene_metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r") as f:
                metadata = json.load(f)

            start_image = input_dir / "random_scene_start.png"
            final_image = input_dir / "random_scene_final.png"

            if start_image.exists() and final_image.exists():
                runs.append({
                    "name": "single_run",
                    "start_image": start_image.name,
                    "final_image": final_image.name,
                    "bucket_hit": metadata["simulation"].get("bucket_hit"),
                    "num_lines": metadata["simulation"].get("num_lines"),
                    "start_ball_x": metadata["simulation"].get("start_ball_x"),
                    "final_ball_x": metadata["simulation"].get("final_ball_x"),
                    "final_ball_y": metadata["simulation"].get("final_ball_y"),
                    "seed": metadata.get("seed"),
                })

    return runs


def generate_html(runs: List[Dict[str, Any]], output_path: pathlib.Path) -> None:
    """Generate HTML summary file."""

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Scene Summary</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .summary-stats {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
        }
        .run-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .run-card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 8px;
        }
        .image-container {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        .image-wrapper {
            flex: 1;
            text-align: center;
        }
        .image-wrapper img {
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .image-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .info {
            margin-top: 10px;
            font-size: 14px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .info-label {
            font-weight: bold;
            color: #555;
        }
        .info-value {
            color: #333;
        }
        .bucket-hit {
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 4px;
            margin-top: 10px;
        }
        .bucket-miss {
            font-size: 18px;
            font-weight: bold;
            color: #f44336;
            text-align: center;
            padding: 10px;
            background-color: #ffebee;
            border-radius: 4px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Random Scene Summary</h1>
"""

    # Add summary statistics
    total_runs = len(runs)
    bucket_hits = [r for r in runs if r["bucket_hit"] is not None]
    hit_rate = (len(bucket_hits) / total_runs * 100) if total_runs > 0 else 0

    html_content += f"""
    <div class="summary-stats">
        <p><strong>Total Runs:</strong> {total_runs}</p>
    </div>

    <div class="container">
"""

    # Add each run
    for run in runs:
        bucket_hit = run["bucket_hit"]
        bucket_class = "bucket-hit" if bucket_hit is not None else "bucket-miss"
        bucket_text = f"Bucket {bucket_hit}" if bucket_hit is not None else "No Bucket Hit"

        html_content += f"""
        <div class="run-card">
            <h3>{run["name"]}</h3>
            <div class="image-container">
                <div class="image-wrapper">
                    <img src="{run['start_image']}" alt="Start">
                    <div class="image-label">Start</div>
                </div>
                <div class="image-wrapper">
                    <img src="{run['final_image']}" alt="Final">
                    <div class="image-label">Final</div>
                </div>
            </div>
            <div class="info">
                <div class="info-row">
                    <span class="info-label">Bucket:</span>
                    <span class="info-value">{bucket_hit if bucket_hit is not None else 'None'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Lines:</span>
                    <span class="info-value">{run['num_lines']}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Start Ball X:</span>
                    <span class="info-value">{run['start_ball_x']:.1f}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Final Ball X:</span>
                    <span class="info-value">{run['final_ball_x']:.1f}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Final Ball Y:</span>
                    <span class="info-value">{run['final_ball_y']:.1f}</span>
                </div>
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    with output_path.open("w") as f:
        f.write(html_content)

    print(f"HTML summary generated: {output_path}")
    print(f"Total runs: {total_runs}")
    print(f"Bucket hits: {len(bucket_hits)} ({hit_rate:.1f}%)")


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"Error: Directory not found: {args.input_dir}")
        return

    # Determine output path
    if args.output is None:
        output_path = args.input_dir / "summary.html"
    else:
        output_path = args.output

    # Collect run data
    runs = collect_run_data(args.input_dir)

    if not runs:
        print(f"No runs found in {args.input_dir}")
        return

    # Generate HTML
    generate_html(runs, output_path)


if __name__ == "__main__":
    main()
