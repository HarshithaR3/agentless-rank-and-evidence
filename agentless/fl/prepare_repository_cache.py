"""Prepare issue-specific repository worktrees and top-three file caches."""

import argparse
import json
import random
import time
from pathlib import Path

from datasets import load_dataset

from agentless.fl.repository_cache import (
    load_issue_cache,
    prepare_issue_cache,
    validate_issue_cache,
)
from agentless.util.utils import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_file", required=True, help="Existing ranked file localization JSONL")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--instance_id", help="Limit preparation to one issue")
    parser.add_argument("--force_instance_id", help="Refresh this issue's cache")
    parser.add_argument("--validate_sample", type=int, default=0, help="Validate this many random successful caches against their original worktree files")
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()

    ranked_rows = load_jsonl(args.start_file)
    ranked_by_id = {row["instance_id"]: row for row in ranked_rows}
    data = load_dataset(args.dataset, split="test")
    dataset_by_id = {row["instance_id"]: row for row in data}
    issue_ids = [args.instance_id] if args.instance_id else list(ranked_by_id)
    summary = {
        "total_issues": len(issue_ids),
        "successfully_cached": [],
        "issues_with_missing_files": [],
        "repository_or_commit_failures": [],
        "incomplete_caches": [],
        "distinct_repositories_cloned": 0,
        "distinct_repository_base_commit_combinations_prepared": 0,
        "repository_preparation_seconds": 0.0,
        "file_extraction_cache_seconds": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    combos = set()
    started = time.monotonic()
    for issue_id in issue_ids:
        if issue_id not in ranked_by_id or issue_id not in dataset_by_id:
            summary["repository_or_commit_failures"].append({"instance_id": issue_id, "error": "missing from RRF output or SWE-bench dataset"})
            continue
        row = ranked_by_id[issue_id]
        issue = dataset_by_id[issue_id]
        try:
            result = prepare_issue_cache(
                issue,
                row["found_files"],
                args.cache_dir,
                force=args.force_instance_id == issue_id,
            )
            combos.add((issue["repo"], issue["base_commit"]))
            summary["repository_preparation_seconds"] += result["repository_prepare_seconds"]
            summary["file_extraction_cache_seconds"] += result["file_cache_seconds"]
            if result["cache_status"] == "complete":
                summary["successfully_cached"].append(issue_id)
            else:
                summary["incomplete_caches"].append(issue_id)
            if result["missing_or_unreadable_files"]:
                summary["issues_with_missing_files"].append({"instance_id": issue_id, "files": result["missing_or_unreadable_files"]})
            if result["cache_hit"]:
                summary["cache_hits"] += 1
            else:
                summary["cache_misses"] += 1
        except Exception as exc:
            summary["repository_or_commit_failures"].append({"instance_id": issue_id, "error": str(exc)})
    repositories_after = {p.name for p in (Path(args.cache_dir) / "repositories").glob("*.git")}
    summary["distinct_repositories_cloned"] = len(repositories_after)
    summary["distinct_repository_base_commit_combinations_prepared"] = len(combos)
    summary["total_elapsed_seconds"] = time.monotonic() - started
    if args.validate_sample:
        candidates = list(summary["successfully_cached"])
        sample = random.Random(args.random_seed).sample(candidates, min(args.validate_sample, len(candidates)))
        validated = []
        for issue_id in sample:
            metadata = load_issue_cache(args.cache_dir, dataset_by_id[issue_id])
            row = ranked_by_id[issue_id]
            validate_issue_cache(metadata, dataset_by_id[issue_id], row["found_files"][:3])
            validated.append(issue_id)
        summary["random_sample_validated"] = validated
    output = Path(args.cache_dir) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
