"""Offline weighted-RRF analysis for existing file localization rankings."""

import argparse
import csv
from pathlib import Path

from agentless.util.utils import load_jsonl
from evaluation.evaluate_file_localization import evaluate, load_ground_truth


DEFAULT_K = 60
DEFAULT_ALPHAS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DEFAULT_GROUND_TRUTH = (
    Path(__file__).resolve().parents[2]
    / "classification"
    / "swebench_lite_classifications.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "weighted_rrf_results.csv"
)


def fuse_weighted_ranked_lists(model_files, retrieval_files, top_n, alpha, k=DEFAULT_K):
    """Return the full ranked union using weighted reciprocal-rank scores."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    if k <= 0:
        raise ValueError("k must be positive")

    scores = {}
    first_seen_order = {}
    weighted_sources = ((model_files, alpha), (retrieval_files, 1 - alpha))

    for ranked_files, weight in weighted_sources:
        seen_in_ranking = set()
        for rank, file_path in enumerate(ranked_files[:top_n], start=1):
            # Duplicates consume their input slot, but contribute only their first rank.
            if file_path in seen_in_ranking:
                continue
            seen_in_ranking.add(file_path)
            scores[file_path] = scores.get(file_path, 0.0) + weight / (k + rank)
            if file_path not in first_seen_order:
                first_seen_order[file_path] = len(first_seen_order)

    ranked_files = sorted(
        scores,
        key=lambda file_path: (-scores[file_path], first_seen_order[file_path]),
    )
    return ranked_files, scores


def load_rankings(path):
    """Load instance-to-ranked-files mappings, rejecting duplicate IDs."""
    rankings = {}
    for row_number, row in enumerate(load_jsonl(path), start=1):
        instance_id = row.get("instance_id")
        files = row.get("found_files")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"Invalid instance_id in {path} row {row_number}")
        if instance_id in rankings:
            raise ValueError(f"Duplicate instance_id {instance_id!r} in {path}")
        if not isinstance(files, list) or any(not isinstance(item, str) for item in files):
            raise ValueError(f"Invalid found_files list in {path} row {row_number}")
        rankings[instance_id] = files
    return rankings


def evaluate_weighted_rrf(
    model_rankings, retrieval_rankings, ground_truth, alphas, top_n, k=DEFAULT_K
):
    """Evaluate each alpha with the repository's existing Recall@3 definition."""
    expected_ids = set(ground_truth)
    if set(model_rankings) != expected_ids:
        raise ValueError("LLM ranking instance IDs do not match ground truth")
    if set(retrieval_rankings) != expected_ids:
        raise ValueError("Embedding ranking instance IDs do not match ground truth")

    results = []
    for alpha in alphas:
        predictions = {}
        for instance_id in sorted(expected_ids):
            ranked_files, _ = fuse_weighted_ranked_lists(
                model_rankings[instance_id],
                retrieval_rankings[instance_id],
                top_n=top_n,
                alpha=alpha,
                k=k,
            )
            predictions[instance_id] = ranked_files

        # This preserves the established macro Recall@3 calculation and patch parsing.
        per_issue = evaluate(ground_truth, predictions)
        if len(per_issue) != len(expected_ids):
            raise ValueError("Not all ground-truth issues were evaluated")
        num_total = len(per_issue)
        recall_at_3 = sum(row["recall_at_3"] for row in per_issue) / num_total
        num_correct = sum(row["num_correct_files"] > 0 for row in per_issue)
        results.append(
            {
                "alpha": alpha,
                "llm_weight": alpha,
                "embedding_weight": 1 - alpha,
                "recall_at_3": recall_at_3,
                "num_correct": num_correct,
                "num_total": num_total,
            }
        )
    return results


def write_results(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "alpha",
        "llm_weight",
        "embedding_weight",
        "recall_at_3",
        "num_correct",
        "num_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate weighted RRF using existing LLM and embedding rankings."
    )
    parser.add_argument("--model-loc-file", required=True, type=Path)
    parser.add_argument("--retrieval-loc-file", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_n", type=int, required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    args = parser.parse_args()

    model_rankings = load_rankings(args.model_loc_file)
    retrieval_rankings = load_rankings(args.retrieval_loc_file)
    ground_truth = load_ground_truth(args.ground_truth)
    results = evaluate_weighted_rrf(
        model_rankings,
        retrieval_rankings,
        ground_truth,
        args.alphas,
        top_n=args.top_n,
        k=args.k,
    )
    write_results(args.output, results)

    print("Alpha | LLM Weight | Embedding Weight | Recall@3 | Correct Issues | Total Issues")
    for row in results:
        print(
            f"{row['alpha']:.2f} | {row['llm_weight']:.0%} | "
            f"{row['embedding_weight']:.0%} | {row['recall_at_3']:.2%} | "
            f"{row['num_correct']} | {row['num_total']}"
        )
    print(f"Results CSV: {args.output}")


if __name__ == "__main__":
    main()
