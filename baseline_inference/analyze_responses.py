"""Analyze Gemini baseline responses against ground-truth bucket labels."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Support both the new <boxed>{value}</boxed> format and legacy LaTeX $\boxed{value}$.
BOXED_TAG_PATTERN = re.compile(r"<boxed>\s*\{([^}]*)\}\s*</boxed>", re.IGNORECASE)
LATEX_BOX_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
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
    """Parse plain-text response logs into structured dictionaries."""
    blocks: List[str] = []
    lines = text.splitlines()
    if not lines:
        return []

    current: List[str] = []
    for line in lines:
        if line.startswith("run_index:") and current:
            chunk = "\n".join(current).strip()
            if chunk:
                blocks.append(chunk)
            current = [line]
        else:
            current.append(line)
    if current:
        chunk = "\n".join(current).strip()
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
                entry["response_text"] = "\n".join(response_lines).strip() if response_lines else None
                if idx < len(lines) and lines[idx].startswith("error:"):
                    error_line = lines[idx]
                    entry["error"] = error_line.split(": ", 1)[1].strip() if ": " in error_line else error_line
                    idx += 1
                continue
            if ": " in line:
                key, value = line.split(": ", 1)
                entry[key.strip()] = value.strip()
            idx += 1
        parsed_blocks.append(entry)
    return parsed_blocks


def _load_raw_entries(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return loaded
    except json.JSONDecodeError:
        pass
    return parse_response_blocks(text)


def parse_responses(path: Path) -> List[ParsedResponse]:
    raw_entries = _load_raw_entries(path)
    parsed: List[ParsedResponse] = []
    for entry in raw_entries:
        run_index = entry.get("run_index")
        try:
            run_index = int(run_index) if run_index is not None else None
        except ValueError:
            run_index = None
        run_dir_value = entry.get("run_dir")
        run_dir = Path(run_dir_value) if run_dir_value else None
        metadata: Dict[str, Any]
        raw_metadata = entry.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata
        else:
            metadata = {
                k[len("metadata."):]: v
                for k, v in entry.items()
                if isinstance(k, str) and k.startswith("metadata.")
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
    match = BOXED_TAG_PATTERN.search(response_text)
    if not match:
        match = LATEX_BOX_PATTERN.search(response_text)
    if not match:
        return None
    value = match.group(1).strip().strip("\"'").lower()
    if value in {"", "none", "null"}:
        return "none"
    return value


def load_ground_truth_info(run_dir: Path) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    metadata_path = run_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return None, None, None
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    simulation = data.get("simulation")
    bucket_hit = None
    if isinstance(simulation, dict):
        bucket_hit = simulation.get("bucket_hit")
    truth = "none" if bucket_hit is None else str(bucket_hit)
    return truth, simulation, str(metadata_path)


def compare_answers(predicted: Optional[str], truth: Optional[str]) -> Optional[bool]:
    if predicted is None or truth is None:
        return None
    return predicted == truth


def default_output_path(responses_path: Path) -> Path:
    parent = responses_path.parent
    results_dir = parent.parent / "results" if parent.name == "responses" else parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"{responses_path.stem}_analysis.json"


def analyze(responses_path: Path) -> Dict[str, Any]:
    parsed = parse_responses(responses_path)
    results: List[Dict[str, Any]] = []
    summary = {
        "total": 0,
        "with_prediction": 0,
        "correct": 0,
        "incorrect": 0,
        "unknown": 0,
    }

    for item in parsed:
        predicted = extract_answer(item.response_text)
        truth, simulation_details, metadata_path = load_ground_truth_info(item.run_dir) if item.run_dir else (None, None, None)
        correct = compare_answers(predicted, truth)

        summary["total"] += 1
        if predicted is not None:
            summary["with_prediction"] += 1
        else:
            summary["unknown"] += 1
        if correct is True:
            summary["correct"] += 1
        elif correct is False:
            summary["incorrect"] += 1

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
                "ground_truth_file": metadata_path,
                "simulation": simulation_details,
            }
        )

    accuracy = None
    if summary["with_prediction"]:
        accuracy = summary["correct"] / summary["with_prediction"]

    return {
        "responses_file": str(responses_path),
        "summary": {
            **summary,
            "accuracy": accuracy,
        },
        "results": results,
    }


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
