"""Run Gemini inference over generated scene runs."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from google import genai
from google.genai import types

DEFAULT_PROMPT = """You are given the start frame of a physics simulation. A ball is dropped from the top of the screen and falls due to gravity. The ball can roll off the lines or the walls in the image. The bouncing of the ball is relatively minor and realistic for normal gravity. Nothing in the image will move besides the ball. Predict which bucket will eventually catch the ball. There are 4 different buckets called bucket 1, bucket 2, bucket 3, and bucket 4. It is also possible that the ball not fall into any of the buckets. 

Please respond with what bucket the ball will fall into. Your final answer should be formatted as "<answer>{bucket number or none}</answer>". For example, if the ball will fall into bucket 2, you should respond with "<answer>2</answer>". If the ball will not fall into any of the buckets, you should respond with "<answer>none</answer>"."""

TIMESTAMP_FMT = "%Y%m%dT%H%M%SZ"


def write_responses(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)

def load_image_bytes(path: Path) -> bytes:
    with path.open("rb") as fp:
        return fp.read()


def gather_run_dirs(batch_dir: Path) -> List[Path]:
    return sorted([p for p in batch_dir.iterdir() if p.is_dir()])


def run_inference_on_dir(
    run_dir: Path,
    client: genai.Client,
    model_name: str,
    prompt: str,
    image_name: str,
    run_index: int,
) -> Dict[str, Any]:
    image_path = run_dir / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image {image_name!r} in {run_dir}")

    image_bytes = load_image_bytes(image_path)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )

    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "image_path": str(image_path),
        "prompt": prompt,
        "response_text": response.text,
        "timestamp": time.time(),
        "metadata": {
            "model_name": model_name,
            "response_type": "baseline_no_sketch",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Gemini inference for physics scenes")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("output/random__batch"),
        help="Directory containing per-run subdirectories",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="random_scene_start.png",
        help="Image filename to send to the model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model identifier",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt to accompany the image",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Number of runs to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline_inference/responses"),
        help="Directory where response JSON files will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = genai.Client()
    collected: List[Dict[str, Any]] = []

    run_dirs = gather_run_dirs(args.batch_dir)
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {args.batch_dir}")

    subset = run_dirs if args.max_runs is None else run_dirs[: args.max_runs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FMT)
    output_path = args.output_dir / f"gemini_responses_{timestamp}.json"

    for idx, run_dir in enumerate(subset, start=1):
        try:
            result = run_inference_on_dir(
                run_dir=run_dir,
                client=client,
                model_name=args.model_name,
                prompt=args.prompt,
                image_name=args.image_name,
                run_index=idx,
            )
        except Exception as exc:
            result = {
                "run_index": idx,
                "run_dir": str(run_dir),
                "prompt": args.prompt,
                "error": str(exc),
                "timestamp": time.time(),
                "metadata": {
                    "model_name": args.model_name,
                    "response_type": "baseline_no_sketch",
                },
            }
        collected.append(result)
        write_responses(output_path, collected)


if __name__ == "__main__":
    main()
