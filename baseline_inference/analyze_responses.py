"""Analyze Gemini baseline responses against ground-truth bucket labels."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ANSWER_PATTERN = re.compile(r"<answer>\s*\{([^}]*)\}\s*</answer>", re.IGNORECASE)
METADATA_FILENAME = "random_scene_metadata.json"


@dataclass
class ParsedResponse:
    run_index: Optional[int]
    run_dir: Optional[Path]
    response_text: Optional[str]
    metadata: Dict[str, Any]
    error: Optional[str]
    raw: Dict[str, Any]


def parse_response_blocks(text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    lines = text.splitlines()
    if not lines:
        return blocks

    current_chunk: List[str] = []
    for line in lines:
        if line.startswith("run_index:") and current_chunk:
            chunk = "
".join(current_chunk).strip()
            if chunk:
                blocks.append(chunk)
            current_chunk = [line]
        else:
            current_chunk.append(line)
    if current_chunk:
        chunk = "
".join(current_chunk).strip()
        if chunk:
            blocks.append(chunk)

    parsed_blocks: List[Dict[str, Any]] = []
    for chunk in blocks:
        entry: Dict[str, Any] = {}
        lines = chunk.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.strip() == "response_text:":
                idx += 1
                response_lines: List[str] = []
                while idx < len(lines) and not lines[idx].startswith("error:"):
                    response_lines.append(lines[idx])
                    idx += 1
                entry["response_text"] = "
".join(response_lines).strip() if response_lines else None
                if idx < len(lines) and lines[idx].startswith("error:"):
                    entry["error"] = lines[idx].split(": ", 1)[1].strip() if ": " in lines[idx] else lines[idx]
                    idx += 1
                continue
            if ": " in line:
                key, value = line.split(": ", 1)
                entry[key.strip()] = value.strip()
            idx += 1
        parsed_blocks.append(entry)
    return parsed_blocks


def parse_responses(path: Path) -> List[ParsedResponse]:
    text = path.read_text(encoding="utf-8")
    raw_entries = parse_response_blocks(text)
    parsed: List[ParsedResponse] = []
    for entry in raw_entries:
        run_index = entry.get("run_index")
        try:
            run_index = int(run_index) if run_index is not None else None
        except ValueError:
            run_index = None
        run_dir_value = entry.get("run_dir")
        run_dir = Path(run_dir_value) if run_dir_value else None
        metadata = {
            k[len("metadata."):]: v
            for k, v in entry.items()
            if k.startswith("metadata.")
        }
        parsed.append(
            ParsedResponse(
                run_index=run_index,
                run_dir=run_dir,
                response_text=entry.get("response_text"),
                metadata=metadata,
                error=entry.get("error"),
                raw=entry,
            )
        )
    return parsed


def extract_answer(response_text: Optional[str]) -> Optional[str]:
    if not response_text:
        return None
    match = ANSWER_PATTERN.search(response_text)
    if not match:
        return None
    value = match.group(1).strip().strip("\"'").lower()
    if value in {"", "none", "null"}:
        return "none"
    return value


def load_ground_truth(run_dir: Path) -> Optional[str]:
    metadata_path = run_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return None
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    bucket_hit = data.get("simulation", {}).get("bucket_hit")
    if bucket_hit is None:
        return "none"
    return str(bucket_hit)


def compare_answers(predicted: Optional[str], truth: Optional[str]) -> Optional[bool]:
    if predicted is None or truth is None:
        return None
    return predicted == truth


def default_output_path(responses_path: Path) -> Path:
    return responses_path.with_name(f"{responses_path.stem}_analysis.json")


def analyze(responses_path: Path) -> Dict[str, Any]:
    parsed = parse_responses(responses_path)
    results: List[Dict[str, Any]] = []
    for item in parsed:
        predicted = extract_answer(item.response_text)
        truth = load_ground_truth(item.run_dir) if item.run_dir else None
        correct = compare_answers(predicted, truth)
        results.append(
            {
                "run_index": item.run_index,
                "run_dir": str(item.run_dir) if item.run_dir else None,
                "model_response": item.response_text,
                "extracted_answer": predicted,
                "ground_truth_answer": truth,
                "correct": correct,
                "error": item.error,
                "metadata": item.metadata or None,
            }
        )
    return {"responses_file": str(responses_path), "results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Gemini baseline responses")
    parser.add_argument(
        "responses_file",
        type=Path,
        help="Path to the baseline response text file (gemini_responses_*.txt)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the analysis JSON (default derives from responses file)",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    analysis = analyze(args.responses_file)
    output_path = args.output_json or default_output_path(args.responses_file)
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
