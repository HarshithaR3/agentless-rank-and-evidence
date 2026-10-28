#!/usr/bin/env python3
"""Evaluate Agentless file-level predictions against SWE-bench gold patches."""

import argparse
import csv
import json
import posixpath
import re
import shlex
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH = (
    REPOSITORY_ROOT / "classification" / "swebench_lite_classifications.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "file_localization_results.csv"
PATCH_PATH_PATTERN = re.compile(r"^diff --git a/(.+) b/(.+)$")


def normalize_path(path: str) -> str:
    """Normalize patch and prediction paths to the same repository-relative form."""
    path = path.strip().replace("\\", "/")
    if path.startswith(("a/", "b/")):
        path = path[2:]
    while path.startswith("./"):
        path = path[2:]
    return posixpath.normpath(path)


def extract_modified_files(patch: str) -> list[str]:
    """Return unique destination paths from git diff headers, in patch order."""
    files = []
    seen = set()
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue

        # Git quotes paths containing whitespace or special characters.
        try:
            fields = shlex.split(line)
        except ValueError as exc:
            raise ValueError(f"Could not parse git diff header {line!r}: {exc}") from exc

        if len(fields) == 4 and fields[2].startswith("a/") and fields[3].startswith("b/"):
            destination = fields[3][2:]
        else:
            # Standard unquoted form: diff --git a/old/path b/new/path.
            match = PATCH_PATH_PATTERN.match(line)
            if not match:
                raise ValueError(f"Unsupported git diff header: {line!r}")
            destination = match.group(2)

        normalized = normalize_path(destination)
        if normalized not in seen:
            seen.add(normalized)
            files.append(normalized)

    if not files:
        raise ValueError("Patch contains no recognizable 'diff --git' file paths")
    return files


def load_ground_truth(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"instance_id", "patch"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Ground-truth CSV is missing required column(s): {', '.join(sorted(missing))}"
            )

        ground_truth = {}
        for row_number, row in enumerate(reader, start=2):
            instance_id = (row.get("instance_id") or "").strip()
            patch = row.get("patch") or ""
            if not instance_id:
                raise ValueError(f"Ground-truth CSV row {row_number} has an empty instance_id")
            if instance_id in ground_truth:
                raise ValueError(f"Duplicate ground-truth instance_id: {instance_id}")
            if not patch.strip():
                raise ValueError(f"Ground-truth patch is empty for {instance_id}")
            try:
                modified_files = extract_modified_files(patch)
            except ValueError as exc:
                raise ValueError(f"Invalid patch for {instance_id}: {exc}") from exc
            ground_truth[instance_id] = {
                "modified_files": modified_files,
                "modified_file_set": set(modified_files),
            }

    return ground_truth


def load_predictions(path: Path) -> dict[str, list[str]]:
    predictions = {}
    with path.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on predictions line {line_number}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Predictions line {line_number} must contain a JSON object")
            if "instance_id" not in record or "found_files" not in record:
                raise ValueError(
                    f"Predictions line {line_number} must contain 'instance_id' and 'found_files'"
                )
            instance_id = record["instance_id"]
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValueError(f"Predictions line {line_number} has an invalid instance_id")
            instance_id = instance_id.strip()
            if instance_id in predictions:
                raise ValueError(f"Duplicate prediction instance_id: {instance_id}")

            found_files = record["found_files"]
            if not isinstance(found_files, list) or any(
                not isinstance(file_path, str) for file_path in found_files
            ):
                raise ValueError(
                    f"'found_files' on predictions line {line_number} must be a list of strings"
                )
            predictions[instance_id] = [normalize_path(p) for p in found_files]

    return predictions


def evaluate(ground_truth: dict[str, dict], predictions: dict[str, list[str]]) -> list[dict]:
    results = []
    for instance_id in sorted(ground_truth.keys() & predictions.keys()):
        ground_truth_files = ground_truth[instance_id]["modified_files"]
        ground_truth_set = ground_truth[instance_id]["modified_file_set"]
        predicted_top3 = predictions[instance_id][:3]
        correct_files = []
        correct_seen = set()
        for file_path in predicted_top3:
            if file_path in ground_truth_set and file_path not in correct_seen:
                correct_seen.add(file_path)
                correct_files.append(file_path)

        num_correct = len(correct_seen)
        recall = num_correct / len(ground_truth_files)
        results.append(
            {
                "instance_id": instance_id,
                "ground_truth_files": json.dumps(ground_truth_files),
                "predicted_top3": json.dumps(predicted_top3),
                "correct_files": json.dumps(correct_files),
                "num_ground_truth_files": len(ground_truth_files),
                "num_correct_files": num_correct,
                "recall_at_3": recall,
            }
        )
    return results


def write_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "instance_id",
        "ground_truth_files",
        "predicted_top3",
        "correct_files",
        "num_ground_truth_files",
        "num_correct_files",
        "recall_at_3",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate top-3 Agentless file localization against gold patch files."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        ground_truth = load_ground_truth(args.ground_truth)
        predictions = load_predictions(args.predictions)
        missing_predictions = sorted(ground_truth.keys() - predictions.keys())
        prediction_only = sorted(predictions.keys() - ground_truth.keys())
        results = evaluate(ground_truth, predictions)
        write_results(args.output, results)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Ground-truth issues: {len(ground_truth)}")
    print(f"Prediction issues: {len(predictions)}")
    print(f"Matching issues: {len(results)}")
    print(f"IDs missing from predictions ({len(missing_predictions)}): {missing_predictions}")
    print(f"IDs present only in predictions ({len(prediction_only)}): {prediction_only}")
    print(f"Number of issues evaluated: {len(results)}")
    if results:
        recall_at_3 = sum(row["recall_at_3"] for row in results) / len(results)
        print(f"Recall@3: {recall_at_3:.2%}")
    else:
        print("Recall@3: N/A (no matching issues)")
    print(f"Per-issue results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
